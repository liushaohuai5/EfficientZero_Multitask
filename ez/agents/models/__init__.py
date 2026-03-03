import torch
import numpy as np
import torch.nn as nn
from ez.utils.format import normalize_state
from ez.utils.format import formalize_obs_lst, DiscreteSupport, allocate_gpu, prepare_obs_lst, symexp, profile
import math


class EfficientZero(nn.Module):
    def __init__(self,
                 representation_model,
                 dynamics_model,
                 reward_prediction_model,
                 value_policy_model,
                 projection_model,
                 projection_head_model,
                 config,
                 sem_model=None,
                 task_emb=None,
                 action_masks=None,
                 **kwargs,
                 ):
        """The basic models in EfficientZero
        Parameters
        ----------
        representation_model: nn.Module
            represent the observations'
        dynamics_model: nn.Module
            dynamics model predicts the next state given the current state and action
        reward_prediction_model: nn.Module
            predict the reward given the next state (Namely, current state and action)
        value_prediction_model: nn.Module
            predict the value given the state
        policy_prediction_model: nn.Module
            predict the policy given the state
        kwargs: dict
            state_norm: bool.
                use state normalization for encoded state
            value_prefix: bool
                predict value prefix instead of reward
        """
        super().__init__()

        self.representation_model = representation_model
        self.sem_model = sem_model
        self.multi_task = config.env.multi_task
        if self.multi_task:
            self.task_emb = task_emb
            self.action_masks = torch.tensor(action_masks).float()
        self.dynamics_model = dynamics_model
        self.reward_prediction_model = reward_prediction_model
        self.value_policy_model = value_policy_model
        self.projection_model = projection_model
        self.projection_head_model = projection_head_model
        self.config = config
        self.continuous_action = config.env.continuous_action
        self.state_norm = kwargs.get('state_norm')
        self.value_prefix = kwargs.get('value_prefix')
        self.v_num = config.train.v_num

    # @profile
    def do_representation(self, obs, task_embedding=None, task_idxs=None):
        if task_embedding is None:
            task_embedding = self.get_task_emb(obs, task_idxs)
        state = self.representation_model(obs, task_embedding)
        if self.state_norm:
            state = normalize_state(state)
        if self.sem_model is not None:
            state = self.sem_model(state)

        return state

    def do_dynamics(self, state, action, task_embedding=None):
        next_state = self.dynamics_model(state, action, task_embedding)
        if self.state_norm:
            next_state = normalize_state(next_state)
        if self.sem_model is not None:
            next_state = self.sem_model(next_state)

        return next_state

    def do_reward_prediction(self, next_state, reward_hidden=None, task_embedding=None):
        # use the predicted state (Namely, current state + action) for reward prediction
        if self.value_prefix:
            value_prefix, reward_hidden = self.reward_prediction_model(next_state, reward_hidden, task_embedding)
            return value_prefix, reward_hidden
        else:
            reward = self.reward_prediction_model(next_state, task_embedding)
            return reward, None

    # @profile
    def do_value_policy_prediction(self, state, task_embedding=None, task_idxs=None):
        value, policy = self.value_policy_model(state, task_embedding)
        if self.multi_task:
            expanded_masks = self.action_masks.cuda()[task_idxs]   # expanded action masks
            if not self.config.env.continuous_action:
                # policy = policy.masked_fill(~expanded_masks.bool(), -1e4)
                policy = policy * expanded_masks - 1e4 * (1 - expanded_masks)
            else:
                mu, std = policy.chunk(2, dim=-1)
                mu = mu * expanded_masks  # mu
                std = std * expanded_masks  # std
                policy = torch.cat([mu, std], dim=1)
        return value, policy

    def do_projection(self, state, with_grad=True):
        # only the branch of proj + pred can share the gradients
        proj = self.projection_model(state)

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head_model(proj)
            return proj
        else:
            return proj.detach()

    def get_task_emb(self, x, task_idxs):
        if self.multi_task:
            task_embedding = self.task_emb(task_idxs)
            if len(x.shape) > 2:    # image based
                task_embed_channels = self.config.model.task_embed_channels
                task_embedding = task_embedding.reshape(x.shape[0], task_embed_channels, 6, 6)
        else:
            task_embedding = None
        return task_embedding

    def get_entropy(self, policy):
        if self.config.env.env == 'Atari':
            policy_prob = torch.softmax(policy, dim=-1)
            H_sample = -(policy_prob * torch.log_softmax(policy, dim=-1)).sum(dim=-1)
        else:
            mean, log_std = policy.chunk(2, dim=-1)
            H_sample = (0.5 * (1.0 + math.log(2 * math.pi)) + log_std).sum(dim=-1)
        return H_sample

    def initial_inference(self, obs, training=False, task_idxs=None, value_reduce='min'):
        task_embedding = self.get_task_emb(obs, task_idxs)
        state = self.do_representation(obs, task_embedding)
        values, policy = self.do_value_policy_prediction(state, task_embedding, task_idxs=task_idxs)

        if training:
            return state, values, policy

        if self.config.model.value_support.type == 'symlog':
            output_values = symexp(values)
        else:
            output_values = DiscreteSupport.vector_to_scalar(values, **self.config.model.value_support)

        value_variance = output_values.var(0)
        if self.continuous_action:
            output_values = output_values.clip(0, 1e5)

        if value_reduce == 'min':
            output_value = output_values[np.random.choice(self.v_num, 2, replace=False)].min(0).values
        else:
            output_value = output_values[np.random.choice(self.v_num, 2, replace=False)].mean(0)

        return state, output_value, policy, value_variance


    def recurrent_inference(self, state, action, reward_hidden, training=False, task_idxs=None, value_reduce='min'):
        task_embedding = self.get_task_emb(state, task_idxs)
        next_state = self.do_dynamics(state, action, task_embedding)
        if self.config.model.reward_parallel:
            if not self.value_prefix:
                value_prefix = self.reward_prediction_model(state, action, task_embedding)
                reward_hidden = None
            else:
                value_prefix, reward_hidden = self.reward_prediction_model(state, action, reward_hidden, task_embedding)
        else:
            value_prefix, reward_hidden = self.do_reward_prediction(next_state, reward_hidden, task_embedding)
        values, policy = self.do_value_policy_prediction(next_state, task_embedding, task_idxs=task_idxs)
        if training:
            return next_state, value_prefix, values, policy, reward_hidden

        if self.config.model.value_support.type == 'symlog':
            output_values = symexp(values)
        else:
            output_values = DiscreteSupport.vector_to_scalar(values, **self.config.model.value_support)

        if self.continuous_action:
            output_values = output_values.clip(0, 1e5)

        value_variance = output_values.var(0)
        if self.config.model.reward_support.type == 'symlog':
            value_prefix = symexp(value_prefix)
        else:
            value_prefix = DiscreteSupport.vector_to_scalar(value_prefix, **self.config.model.reward_support)

        if value_reduce == 'min':
            output_value = output_values[np.random.choice(self.v_num, 2, replace=False)].min(0).values
        else:
            output_value = output_values[np.random.choice(self.v_num, 2, replace=False)].mean(0)

        return next_state, value_prefix, output_value, policy, reward_hidden, value_variance

    def get_weights(self, part='none'):
        if part == 'reward':
            weights = self.reward_prediction_model.state_dict()
        else:
            weights = self.state_dict()

        return {k: v.cpu() for k, v in weights.items()}

    def set_weights(self, weights, hard=False):
        if hard:
            self.load_state_dict(weights)
        else:
            # EMA update
            tau = 0.01
            self.load_state_dict({
                k: v * (1-tau) + weights[k].to(v.device) * tau for k, v in self.state_dict().items()
            })


    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)