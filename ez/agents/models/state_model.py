import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np
from ez.agents.models.layer import RunningMeanStd, ImproveResidualBlock, mlp, act_function


# Encode the observations into hidden states
class RepresentationNetwork(nn.Module):
    def __init__(
        self,
        observation_shape,
        n_stacked_obs,
        num_blocks,
        rep_net_shape,
        hidden_shape,
        task_embed_dim,
        use_bn=True,
        multi_task=False,
    ):
        """Representation network
        Parameters
        ----------
        observation_shape: tuple or list
            shape of observations: [C, W, H]
        n_stacked_obs: int
            number of stacked observation
        num_blocks: int
            number of res blocks
        rep_net_shape: int
            shape of hidden layers
        hidden_shape:
            dim of output hidden state
        use_bn: bool
            True -> Batch normalization
        """
        super().__init__()
        # print('obs shape: {}'.format(observation_shape))
        # print('stacked obs: {}
        self.multi_task = multi_task
        if not multi_task:
            task_embed_dim = 0
        self.running_mean_std = RunningMeanStd(observation_shape * n_stacked_obs)
        self.mlp = mlp(observation_shape * n_stacked_obs + task_embed_dim, [], hidden_shape, output_activation=nn.Tanh)
        self.Rep_resblocks = nn.ModuleList(
            [ImproveResidualBlock(hidden_shape, rep_net_shape) for _ in range(num_blocks)]
        )

    def forward(self, x, task_embedding=None):
        # print('input shape: {}'.format(x.shape))

        # pre process
        x = self.running_mean_std(x)
        if self.multi_task:
            x = torch.cat([x, task_embedding], dim=-1)
        x = self.mlp(x)
        # res block
        for block in self.Rep_resblocks:
            x = block(x)

        return x

    def get_param_mean(self):
        mean = []
        for name, param in self.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)
        return mean


# Predict next hidden states given current states and actions
class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        hidden_shape,
        action_shape,
        num_blocks,
        dyn_shape,
        act_embed_shape,
        task_embed_dim,
        rew_net_shape,
        reward_support_size,
        init_zero=False,
        use_bn=True,
        action_embedding=False,
        multi_task=False,
    ):
        """Dynamics network
        Parameters
        ----------
        hidden_shape: int
            dim of input hidden state
        action_shape: int
            dim of action
        num_blocks: int
            number of res blocks
        dyn_shape: int
            number of nodes of hidden layer
        act_embed_shape: int
            dim of action embedding
        rew_net_shape: list
            hidden layers of the reward prediction head (MLP head)
        reward_support_size: int
            dim of reward output
        init_zero: bool
            True -> zero initialization for the last layer of reward mlp
        use_bn: bool
            True -> Batch normalization
        """
        super().__init__()
        self.hidden_shape = hidden_shape
        self.action_embedding = action_embedding
        self.multi_task = multi_task
        if not multi_task:
            task_embed_dim = 0

        if action_embedding:
            self.act_mlp = mlp(action_shape + task_embed_dim, [], act_embed_shape, output_activation=nn.Tanh)
            self.dyn_ln1 = nn.LayerNorm(hidden_shape + task_embed_dim + act_embed_shape)
            self.dyn_linear1 = nn.Linear(hidden_shape + task_embed_dim + act_embed_shape, dyn_shape)
        else:
            self.dyn_ln1 = nn.LayerNorm(hidden_shape + task_embed_dim + action_shape)
            self.dyn_linear1 = nn.Linear(hidden_shape + task_embed_dim + action_shape, dyn_shape)

        self.dyn_linear2 = nn.Linear(dyn_shape, hidden_shape)

        if num_blocks > 0:
            self.dyn_resblocks = nn.ModuleList(
                [ImproveResidualBlock(hidden_shape, dyn_shape) for _ in range(num_blocks)]
            )
        else:
            self.dyn_resblocks = nn.ModuleList([])


    def forward(self, hidden, action, task_embedding):

        # action embedding
        if self.action_embedding:
            if self.multi_task:
                action_input = torch.cat((action, task_embedding), dim=-1)
            else:
                action_input = action
            act_emb = self.act_mlp(action_input)
        else:
            act_emb = action
        if self.multi_task:
            x = torch.cat((hidden, task_embedding, act_emb), dim=-1)
        else:
            x = torch.cat((hidden, act_emb), dim=-1)
        x = self.dyn_ln1(x)
        x = self.dyn_linear1(x)
        x = act_function(x)

        x = self.dyn_linear2(x)

        state = hidden + x

        # residual tower for dynamic model (2nd -> num blocks)
        for block in self.dyn_resblocks:
            state = block(state)

        return state

    def get_dynamic_mean(self):

        mean = []
        for name, param in self.dyn_linear1.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)

        return mean


class RewardNetworkParallel(nn.Module):
    def __init__(
        self,
        hidden_shape,
        action_shape,
        act_embed_shape,
        task_embed_dim,
        rew_net_shape,
        reward_support_size,
        num_blocks,
        init_zero=False,
        use_bn=True,
        action_embedding=False,
        multi_task=False,
    ):
        super().__init__()
        self.hidden_shape = hidden_shape
        self.rew_net_shape = rew_net_shape
        self.reward_support_size = reward_support_size
        self.action_embedding = action_embedding
        self.multi_task = multi_task
        if not multi_task:
            task_embed_dim = 0

        if action_embedding:
            self.act_mlp = mlp(action_shape + task_embed_dim, [], act_embed_shape, output_activation=nn.Tanh)
            self.dyn_ln1 = nn.LayerNorm(hidden_shape + task_embed_dim + act_embed_shape)
            self.dyn_linear1 = nn.Linear(hidden_shape + task_embed_dim + act_embed_shape, hidden_shape)
        else:
            self.dyn_ln1 = nn.LayerNorm(hidden_shape + task_embed_dim + action_shape)
            self.dyn_linear1 = nn.Linear(hidden_shape + task_embed_dim + action_shape, hidden_shape)

        self.dyn_linear2 = nn.Linear(hidden_shape, hidden_shape)

        if num_blocks > 0:
            self.dyn_resblocks = nn.ModuleList(
                [ImproveResidualBlock(hidden_shape, hidden_shape) for _ in range(num_blocks)]
            )
        else:
            self.dyn_resblocks = nn.ModuleList([])

        self.rew_net = mlp(self.hidden_shape, self.rew_net_shape, self.reward_support_size,
                           init_zero=init_zero,
                           use_bn=use_bn,
                           norm_type='batchnorm'
                           )

    def forward(self, hidden, action, task_embedding, reward_hidden=None):

        # action embedding
        if self.action_embedding:
            if self.multi_task:
                action_input = torch.cat((action, task_embedding), dim=-1)
            else:
                action_input = action
            act_emb = self.act_mlp(action_input)
        else:
            act_emb = action
        if self.multi_task:
            x = torch.cat((hidden, task_embedding, act_emb), dim=-1)
        else:
            x = torch.cat((hidden, act_emb), dim=-1)
        x = self.dyn_ln1(x)
        x = self.dyn_linear1(x)
        x = act_function(x)

        x = self.dyn_linear2(x)

        state = hidden + x

        # residual tower for dynamic model (2nd -> num blocks)
        for block in self.dyn_resblocks:
            state = block(state)

        reward = self.rew_net(state)
        return reward

class RewardNetwork(nn.Module):
    def __init__(
        self,
        hidden_shape,
        task_embed_dim,
        rew_net_shape,
        reward_support_size,
        num_blocks,
        init_zero=False,
        use_bn=True,
        multi_task=False,
    ):
        super().__init__()
        self.hidden_shape = hidden_shape
        self.multi_task = multi_task
        if not multi_task:
            task_embed_dim = 0
        self.rew_net_shape = rew_net_shape
        self.reward_support_size = reward_support_size
        self.rew_resblocks = nn.ModuleList([
            ImproveResidualBlock(self.hidden_shape + task_embed_dim, self.hidden_shape + task_embed_dim) for _ in range(num_blocks)
        ])
        self.ln = nn.LayerNorm(self.hidden_shape + task_embed_dim)
        self.rew_net = mlp(self.hidden_shape + task_embed_dim, self.rew_net_shape, self.reward_support_size,
                           init_zero=init_zero,
                           use_bn=use_bn)

    def forward(self, next_state, task_embedding, reward_hidden=None):
        if self.multi_task:
            next_state = torch.cat((next_state, task_embedding), dim=-1)
        # next_state = self.rew_resblock(next_state)
        for block in self.rew_resblocks:
            next_state = block(next_state)
        next_state = self.ln(next_state)
        reward = self.rew_net(next_state)
        return reward


class ValueNetwork(nn.Module):
    def __init__(
        self,
        hidden_shape,
        task_embed_dim,
        val_net_shape,
        full_support_size,
        num_blocks,
        init_zero=False,
        use_bn=True,
        multi_task=False,
    ):
        super().__init__()
        self.hidden_shape = hidden_shape
        self.val_net_shape = val_net_shape
        self.multi_task = multi_task
        if not multi_task:
            task_embed_dim = 0

        self.val_resblocks = nn.ModuleList([
            ImproveResidualBlock(hidden_shape + task_embed_dim, hidden_shape + task_embed_dim) for _ in range(num_blocks)
        ])
        self.val_ln = nn.LayerNorm(hidden_shape + task_embed_dim)
        self.val_net = mlp(hidden_shape + task_embed_dim, self.val_net_shape, full_support_size,
                           # init_zero=init_zero,
                           use_bn=use_bn,
                           # dropout=0.01,
                           norm_type='batchnorm'
                           )

    def forward(self, x, task_embedding):
        if self.multi_task:
            x = torch.cat((x, task_embedding), dim=-1)
        for resblock in self.val_resblocks:
            x = resblock(x)
        x = self.val_ln(x)
        value = self.val_net(x)
        return value


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        hidden_shape,
        task_embed_dim,
        pi_net_shape,
        action_shape,
        num_blocks,
        init_zero=False,
        use_bn=True,
        p_norm=False,
        multi_task=False,
    ):
        super().__init__()
        self.hidden_shape = hidden_shape
        self.pi_net_shape = pi_net_shape
        self.action_shape = action_shape
        self.multi_task = multi_task
        if not multi_task:
            task_embed_dim = 0

        self.pi_resblocks = nn.ModuleList([
            ImproveResidualBlock(hidden_shape + task_embed_dim, hidden_shape + task_embed_dim) for _ in range(1)
        ])
        self.pi_ln = nn.LayerNorm(hidden_shape + task_embed_dim)
        self.pi_net = mlp(hidden_shape + task_embed_dim, pi_net_shape, action_shape * 2,
                          init_zero=init_zero,
                          use_bn=use_bn,
                          p_norm=p_norm,
                          norm_type='batchnorm'
                          )

    def forward(self, x, task_embedding):
        if self.multi_task:
            x = torch.cat([x, task_embedding], dim=-1)
        for resblock in self.pi_resblocks:
            x = resblock(x)
        x = self.pi_ln(x)
        policy = self.pi_net(x)

        return policy


class ValuePolicyNetworkSplit(nn.Module):
    def __init__(
        self,
        hidden_shape,
        task_embed_dim,
        val_net_shape,
        pi_net_shape,
        action_shape,
        full_support_size,
        num_blocks,
        init_zero=False,
        use_bn=True,
        p_norm=False,
        policy_distr='squashed_gaussian',
        value_support=None,
        multi_task=False,
        **kwargs
    ):
        super().__init__()
        self.v_num = kwargs.get('v_num')
        self.multi_task = multi_task
        if not multi_task:
            task_embed_dim = 0

        self.value_networks = nn.ModuleList([
            ValueNetwork(hidden_shape, task_embed_dim, val_net_shape, full_support_size, num_blocks, init_zero, use_bn, multi_task=multi_task) for _ in range(self.v_num)
        ])
        self.policy_network = PolicyNetwork(hidden_shape, task_embed_dim, pi_net_shape, action_shape, num_blocks, init_zero, use_bn, p_norm, multi_task=multi_task)
        self.value_support = value_support
        self.value_policy_detach = kwargs.get('value_policy_detach')
        self.policy_distr = policy_distr
        self.init_std = 1.0
        self.min_std = 0.1

    def forward(self, x, task_embedding):
        if self.value_policy_detach:
            x = x.detach()  # for decoupled training

        values = []
        for value_network in self.value_networks:
            values.append(value_network(x, task_embedding))
        values = torch.stack(values)
        policy = self.policy_network(x, task_embedding)

        action_space_size = policy.shape[-1] // 2
        if self.policy_distr == 'squashed_gaussian':
            policy[:, :action_space_size] = 5 * torch.tanh(policy[:, :action_space_size] / 5)  # soft clamp mu
            policy[:, action_space_size:] = torch.nn.functional.softplus(
                policy[:, action_space_size:] + self.init_std) + self.min_std  # same as Dreamer-v3

        return values, policy


# predict the value and policy given hidden states
class ValuePolicyNetwork(nn.Module):
    def __init__(
        self,
        hidden_shape,
        task_embed_dim,
        val_net_shape,
        pi_net_shape,
        action_shape,
        full_support_size,
        num_blocks,
        init_zero=False,
        use_bn=True,
        p_norm=False,
        policy_distr='squashed_gaussian',
        value_support=None,
        multi_task=False,
        **kwargs
    ):
        super().__init__()
        self.v_num = kwargs.get('v_num')
        self.hidden_shape = hidden_shape
        self.val_net_shape = val_net_shape
        self.action_shape = action_shape
        self.pi_net_shape = pi_net_shape

        self.action_space_size = action_shape
        self.policy_distr = policy_distr
        self.multi_task = multi_task
        self.policy_sem_V = kwargs.get('policy_sem_V')
        if not multi_task:
            task_embed_dim = 0

        self.resblocks = nn.ModuleList([
            ImproveResidualBlock(hidden_shape + task_embed_dim, hidden_shape + task_embed_dim) for _ in range(num_blocks)
        ])
        self.ln = nn.LayerNorm(hidden_shape + task_embed_dim)

        self.val_nets = nn.ModuleList([
            mlp(self.hidden_shape + task_embed_dim, self.val_net_shape, full_support_size,
                # init_zero=init_zero,    # TODO: try if works using value zero_init in humanoid bench
                use_bn=use_bn,
                # dropout=0.01,
                norm_type='batchnorm'
            )
            for _ in range(self.v_num)])
        self.pi_net = mlp(
            self.hidden_shape + task_embed_dim, self.pi_net_shape, self.action_shape * 2,
            init_zero=init_zero,
            use_bn=use_bn,
            # p_norm=p_norm,
            norm_type='batchnorm'
        )

        self.value_support = value_support
        self.value_policy_detach = kwargs.get('value_policy_detach')
        self.init_std = 1.0
        self.min_std = 0.1

    def forward(self, x, task_embedding):
        if self.multi_task:
            x = torch.cat((x, task_embedding), dim=-1)
        if self.value_policy_detach:
            x = x.detach()  # for decoupled training

        # x = self.resblock(x)
        for block in self.resblocks:
            x = block(x)
        x = self.ln(x)
        values = []
        for val_net in self.val_nets:
            values.append(val_net(x))
        values = torch.stack(values)
        policy = self.pi_net(x)

        action_space_size = policy.shape[-1] // 2
        if self.policy_distr == 'squashed_gaussian':
            policy[:, :action_space_size] = 5 * torch.tanh(policy[:, :action_space_size] / 5)  # soft clamp mu
            policy[:, action_space_size:] = torch.nn.functional.softplus(
                policy[:, action_space_size:] + self.init_std) + self.min_std  # same as Dreamer-v3

        return values, policy
