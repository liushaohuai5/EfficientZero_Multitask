import time
import copy
import math
import torch
import torch.nn as nn
from ez.agents.base import Agent
from omegaconf import open_dict

import numpy as np
from ez.envs import make_dmc, make_envs
from ez.utils.format import DiscreteSupport, calc_horizon
from ez.agents.models import EfficientZero
from ez.agents.models.state_model import RepresentationNetwork, DynamicsNetwork, RewardNetwork, RewardNetworkParallel, ValuePolicyNetworkSplit, ValuePolicyNetwork
from ez.agents.models.layer import SEM

act_function = nn.functional.relu
activation = nn.ReLU

class EZDMCStateAgent(Agent):
    def __init__(self, config):
        super().__init__(config)

        self.update_config()

        self.num_blocks = config.model.num_blocks
        self.fc_layers = config.model.fc_layers
        self.down_sample = config.model.down_sample
        self.state_norm = config.model.state_norm
        self.value_prefix = config.model.value_prefix
        self.init_zero = config.model.init_zero
        self.value_policy_detach = config.train.value_policy_detach
        self.v_num = config.train.v_num
        self.multi_task = config.env.multi_task

    def update_config(self):
        assert not self._update
        env = make_dmc(self.config.env.game, seed=self.config.env.base_seed, save_path=None, **self.config.env)
        obs = env.reset()
        obs_shape = obs.shape[0]
        action_space_size = env.action_space.shape[0]
        if self.multi_task:
            with open_dict(self.config):
                if self.config.env.difficulty == 'medium':
                    self.config.env.env_names = [
                        'walker_stand', 'walker_walk', 'walker_run', 'cartpole_swingup_sparse', 'acrobot_swingup', 'cheetah_run', 'hopper_stand', 'hopper_hop'
                    ]
                elif self.config.env.difficulty == 'hard':
                    self.config.env.env_names = [
                        'humanoid_stand', 'humanoid_walk', 'humanoid_run', 'dog_stand', 'dog_walk', 'dog_trot', 'dog_run'
                    ]
            envs = make_envs(self.config.env.env, self.config.env.game, self.config.data.num_envs, seed=self.config.env.base_seed,
                             **self.config.env)
            action_space_size = int(max([env.action_space.shape[0] for env in envs]))
            obs_shape = int(max(env.reset().shape[0] for env in envs))

        reward_support = DiscreteSupport(self.config)
        reward_size = reward_support.size
        self.reward_support = reward_support

        value_support = DiscreteSupport(self.config)
        value_size = value_support.size
        self.value_support = value_support

        localtime = time.strftime('%Y-%m-%d %H:%M:%S')
        tag = '{}-seed={}-{}/'.format(self.config.tag, self.config.env.base_seed, localtime)

        with open_dict(self.config):
            self.config.env.action_space_size = action_space_size
            self.config.env.obs_shape = obs_shape
            self.config.rl.discount **= self.config.env.n_skip
            self.config.model.reward_support.size = reward_size
            self.config.model.value_support.size = value_size
            if 'finger' in self.config.env.game:
                print('game is finger turn, use small std noise !!!!!!!')
                self.config.mcts.std_magnification = 2
            self.config.save_path += tag
            self.config.optimizer.lr_decay_steps = int(self.config.train.training_steps) * 0.9
            self.config.mcts.max_horizon = calc_horizon(self.config.mcts.num_simulations, self.config.mcts.num_top_actions)

            self.config.env.task_num = 1
            if self.config.env.multi_task:
                mask = np.zeros((len(envs), action_space_size)).astype(np.float32)
                for i, env in enumerate(envs):
                    mask[i, :env.action_space.shape[0]] = 1.
                self.config.env.action_masks = mask.tolist()
                self.config.env.task_num = len(envs)
                mini_batch_size = self.config.train.batch_size // len(envs)
                self.config.train.batch_size = int(mini_batch_size * len(envs))

        self.action_space_size = self.config.env.action_space_size
        self.obs_shape = self.config.env.obs_shape
        self.n_stack = self.config.env.n_stack
        self.rep_net_shape = self.config.model.rep_net_shape
        self.hidden_shape = self.config.model.hidden_shape
        self.task_embedding_shape = self.config.model.task_embedding_shape
        self.dyn_shape = self.config.model.dyn_shape
        self.act_embed_shape = self.config.model.act_embed_shape
        self.rew_net_shape = self.config.model.rew_net_shape
        self.val_net_shape = self.config.model.val_net_shape
        self.pi_net_shape = self.config.model.pi_net_shape

        self.proj_hid_shape = self.config.model.proj_hid_shape
        self.pred_hid_shape = self.config.model.pred_hid_shape
        self.proj_shape = self.config.model.proj_shape
        self.pred_shape = self.config.model.pred_shape

        self._update = True
        self.use_bn = self.config.model.use_bn
        self.use_p_norm = self.config.model.use_p_norm

    def build_model(self):

        task_embedding_choice = self.config.model.task_embedding_choice
        representation_model = RepresentationNetwork(self.obs_shape, self.n_stack, self.num_blocks,
                                                     self.rep_net_shape, self.hidden_shape, task_embed_dim=self.task_embedding_shape,
                                                     use_bn=self.use_bn, multi_task=self.multi_task)
        value_output_size = self.config.model.value_support.size if self.config.model.value_support.type != 'symlog' else 1
        reward_output_size = self.config.model.reward_support.size if self.config.model.reward_support.type != 'symlog' else 1
        dynamics_model = DynamicsNetwork(self.hidden_shape, self.action_space_size, self.num_blocks, self.dyn_shape,
                                         self.act_embed_shape, self.task_embedding_shape, self.rew_net_shape, reward_output_size,
                                         use_bn=self.use_bn, multi_task=self.multi_task)
        VP_model = ValuePolicyNetworkSplit if self.config.train.use_policy_gradient else ValuePolicyNetwork
        value_policy_model = VP_model(
            self.hidden_shape, self.task_embedding_shape, self.val_net_shape, self.pi_net_shape,
            self.action_space_size, value_output_size, self.num_blocks, init_zero=self.init_zero,
            use_bn=self.use_bn, p_norm=self.use_p_norm, policy_distr=self.config.model.policy_distribution,
            value_support=self.config.model.value_support, value_policy_detach=self.value_policy_detach,
            v_num=self.v_num, multi_task=self.multi_task,
        )

        if self.config.env.multi_task:
            self.task_emb = nn.Embedding(self.config.env.task_num, self.task_embedding_shape, max_norm=1)
        else:
            self.task_emb = None

        if self.config.model.reward_parallel:
            reward_prediction_model = RewardNetworkParallel(
                self.hidden_shape, self.action_space_size, self.act_embed_shape, self.task_embedding_shape,
                self.rew_net_shape, reward_output_size, self.num_blocks, init_zero=self.init_zero, use_bn=self.use_bn,
                action_embedding=self.config.model.action_embedding, multi_task=self.multi_task
            )
        else:
            reward_prediction_model = RewardNetwork(
                self.hidden_shape, self.task_embedding_shape,
                self.rew_net_shape, reward_output_size, self.num_blocks, init_zero=self.init_zero, use_bn=self.use_bn,
                multi_task=self.multi_task
            )

        projection_model = nn.Sequential(
            nn.Linear(self.hidden_shape, self.proj_hid_shape),
            nn.LayerNorm(self.proj_hid_shape),
            nn.ReLU(),
            nn.Linear(self.proj_hid_shape, self.proj_hid_shape),
            nn.LayerNorm(self.proj_hid_shape),
            nn.ReLU(),
            nn.Linear(self.proj_hid_shape, self.proj_shape),
            nn.LayerNorm(self.proj_shape)
        )
        projection_head_model = nn.Sequential(
            nn.Linear(self.proj_shape, self.pred_hid_shape),
            nn.LayerNorm(self.pred_hid_shape),
            nn.ReLU(),
            nn.Linear(self.pred_hid_shape, self.pred_shape),
        )

        if self.config.model.use_sem:
            sem_model = SEM(V=self.config.model.sem.V)
        else:
            sem_model = None

        ez_model = EfficientZero(representation_model, dynamics_model, reward_prediction_model, value_policy_model,
                                 projection_model, projection_head_model, self.config,
                                 state_norm=self.state_norm, value_prefix=self.value_prefix,
                                 sem_model=sem_model,
                                 action_masks=self.config.env.action_masks if self.config.env.multi_task else None,
                                 task_emb=self.task_emb)

        return ez_model
