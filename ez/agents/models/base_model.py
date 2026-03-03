import torch
import math
import torch.nn as nn
import numpy as np
from .layer import ResidualBlock, conv3x3, mlp


# Down_sample observations before representation network (See paper appendix Network Architecture)
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.resblocks1 = nn.ModuleList(
            [ResidualBlock(out_channels // 2, out_channels // 2) for _ in range(1)]
        )
        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.downsample_block = ResidualBlock(out_channels // 2, out_channels, downsample=self.conv2, stride=2)
        self.resblocks2 = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels) for _ in range(1)]
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels) for _ in range(1)]
        )
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


# Encode the observations into hidden states
class RepresentationNetwork(nn.Module):
    def __init__(self, observation_shape, num_blocks, num_channels, downsample, multi_task=False, task_embed_channels=16):
        """
        Representation network
        :param observation_shape: tuple or list, shape of observations: [C, W, H]
        :param num_blocks: int, number of res blocks
        :param num_channels: int, channels of hidden states
        :param downsample: bool, True -> do downsampling for observations. (For board games, do not need)
        """
        super().__init__()
        self.downsample = downsample
        self.multi_task = multi_task
        if not multi_task:
            task_embed_channels = 0
        if self.downsample:
            self.downsample_net = DownSample(
                observation_shape[0],
                num_channels,
            )
        else:
            self.conv = conv3x3(
                observation_shape[0],
                num_channels,
            )
            self.bn = nn.BatchNorm2d(num_channels)
        if self.multi_task:
            self.conv1 = conv3x3(
                num_channels + task_embed_channels, num_channels
            )
            self.bn1 = nn.BatchNorm2d(num_channels)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x, task_embedding=None):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = nn.functional.relu(x)

        if self.multi_task:
            x = torch.cat([x, task_embedding], dim=1)
            x = self.conv1(x)
            x = self.bn1(x)
            x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)

        return x


# Predict next hidden states given current states and actions
class DynamicsNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, action_space_size, is_continuous=False,
                 action_embedding=False, action_embedding_dim=32,
                 multi_task=False, task_embed_channels=16):
        """
        Dynamics network
        :param num_blocks: int, number of res blocks
        :param num_channels: int, channels of hidden states
        :param action_space_size: int, action space size
        """
        super().__init__()
        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        self.num_channels = num_channels
        self.action_space_size = action_space_size
        self.multi_task = multi_task
        if not multi_task:
            task_embed_channels = 0

        action_space_dim = action_space_size if is_continuous else 1

        if action_embedding:
            self.conv1x1 = nn.Conv2d(action_space_dim + task_embed_channels, self.action_embedding_dim, 1)
            self.ln = nn.LayerNorm([action_embedding_dim, 6, 6])
            self.conv = conv3x3(num_channels + task_embed_channels + self.action_embedding_dim, num_channels)
        else:
            self.conv = conv3x3(num_channels + task_embed_channels + action_space_dim, num_channels)

        self.bn = nn.BatchNorm2d(num_channels)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

    def forward(self, state, action, task_embedding=None):
        # encode action
        if not self.is_continuous:
            action_place = torch.ones((
                state.shape[0],
                1,
                state.shape[2],
                state.shape[3],
            )).cuda().float()

            action_place = (
                    action[:, :, None, None] * action_place / self.action_space_size
            )
        else:
            action_place = action.reshape(*action.shape, 1, 1).repeat(1, 1, state.shape[-2], state.shape[-1])

        if self.action_embedding:
            if self.multi_task:
                action_input = torch.cat([action_place, task_embedding], dim=1)
            else:
                action_input = action_place
            action_place = self.conv1x1(action_input)
            action_place = self.ln(action_place)
            action_place = nn.functional.relu(action_place)

        if self.multi_task:
            x = torch.cat([state, task_embedding], dim=1)
        else:
            x = state
        x = torch.cat((x, action_place), dim=1)
        x = self.conv(x)
        x = self.bn(x)

        x = x + state
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        state = x

        return state

class PolicyNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, reduced_channels, flatten_size, fc_layers,
                 output_size, init_zero, is_continuous=False, policy_distribution='beta',
                 multi_task=False, task_embed_channels=16):
        super().__init__()
        self.is_continuous = is_continuous
        self.policy_distr = policy_distribution
        self.multi_task = multi_task
        if not multi_task:
            task_embed_channels = 0

        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels + task_embed_channels, num_channels + task_embed_channels) for _ in range(num_blocks)]
        )
        self.bn1 = nn.BatchNorm2d(num_channels + task_embed_channels)
        self.conv1x1 = nn.Conv2d(num_channels + task_embed_channels, reduced_channels, 1)
        self.bn = nn.BatchNorm2d(reduced_channels)
        self.block_output_size = flatten_size
        self.fc = mlp(flatten_size, fc_layers, output_size, init_zero=init_zero)

        self.init_std = 1.0
        self.min_std = 0.1

    def forward(self, x, task_embedding=None):
        if self.multi_task:
            x = torch.cat([x, task_embedding], dim=1)
        for block in self.resblocks:
            x = block(x)
        policy = self.conv1x1(x)
        policy = self.bn(policy)
        policy = nn.functional.relu(policy)
        # policy = nn.functional.avg_pool2d(x, 2)
        policy = policy.reshape(-1, self.block_output_size)
        policy = self.fc(policy)

        if self.is_continuous:
            action_space_size = policy.shape[-1] // 2
            if self.policy_distr == 'squashed_gaussian':
                policy[:, :action_space_size] = 5 * torch.tanh(policy[:, :action_space_size] / 5)  # soft clamp mu
                policy[:, action_space_size:] = torch.nn.functional.softplus(policy[:, action_space_size:] + self.init_std) + self.min_std  # same as Dreamer-v3
            elif self.policy_distr == 'truncated_gaussian':
                policy[:, action_space_size:] = torch.nn.functional.softplus(policy[:, action_space_size:])     # force std > 0
            elif self.policy_distr == 'beta':
                policy = torch.nn.functional.softplus(policy) + 1.0     # force alpha, beta >= 1
            else:
                raise NotImplementedError

        return policy

class ValueNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, reduced_channels, flatten_size, fc_layers,
                 output_size, init_zero, multi_task=False, task_embed_channels=16):
        super().__init__()
        self.multi_task = multi_task
        if not multi_task:
            task_embed_channels = 0
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels + task_embed_channels, num_channels + task_embed_channels) for _ in range(num_blocks)]
        )
        self.bn1 = nn.BatchNorm2d(num_channels + task_embed_channels)
        self.conv1x1 = nn.Conv2d(num_channels + task_embed_channels, reduced_channels, 1)
        self.bn = nn.BatchNorm2d(reduced_channels)
        self.block_output_size = flatten_size
        self.fc = mlp(flatten_size, fc_layers, output_size, init_zero=init_zero)

    def forward(self, x, task_embedding=None):
        if self.multi_task:
            x = torch.cat([x, task_embedding], dim=1)
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1(x)
        value = self.bn(value)
        value = nn.functional.relu(value)
        # value = nn.functional.avg_pool2d(x, 2)
        value = value.reshape(-1, self.block_output_size)
        value = self.fc(value)

        return value


class ValuePolicyNetworkSplit(nn.Module):
    def __init__(self, num_blocks, num_channels, reduced_channels, flatten_size, fc_layers, value_output_size,
                 policy_output_size, init_zero, is_continuous=False, policy_distribution='beta',
                 multi_task=False, task_embed_channels=16, **kwargs):
        super().__init__()
        self.multi_task = multi_task
        if not multi_task:
            task_embed_channels = 0
        self.v_num = kwargs.get('v_num')
        self.value_policy_detach = kwargs.get('value_policy_detach')
        self.value_networks = nn.ModuleList([
            ValueNetwork(num_blocks, num_channels, reduced_channels, flatten_size, fc_layers, value_output_size, init_zero if not is_continuous else False,
                         multi_task=multi_task, task_embed_channels=task_embed_channels)
        for _ in range(self.v_num)])
        self.policy_network = PolicyNetwork(num_blocks, num_channels, reduced_channels, flatten_size, fc_layers, policy_output_size, init_zero, is_continuous, policy_distribution,
                                            multi_task=multi_task, task_embed_channels=task_embed_channels)

    def forward(self, x, task_embedding=None):
        if self.value_policy_detach:
            x = x.detach()  # try decoupled training

        values = []
        for value_network in self.value_networks:
            values.append(value_network(x, task_embedding))
        values = torch.stack(values)
        policy = self.policy_network(x, task_embedding)

        return values, policy

class ValuePolicyNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, reduced_channels, flatten_size, fc_layers, value_output_size,
                 policy_output_size, init_zero, is_continuous=False, policy_distribution='beta',
                 multi_task=False, task_embed_channels=16, **kwargs):
        super().__init__()
        self.v_num = kwargs.get('v_num')
        self.multi_task = multi_task
        if not multi_task:
            task_embed_channels = 0
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels + task_embed_channels, num_channels + task_embed_channels) for _ in range(num_blocks)]
        )
        self.conv1x1_values = nn.ModuleList([nn.Conv2d(num_channels + task_embed_channels, reduced_channels, 1) for _ in range(self.v_num)])
        self.conv1x1_policy = nn.Conv2d(num_channels + task_embed_channels, reduced_channels, 1)
        self.bn_values = nn.ModuleList([nn.BatchNorm2d(reduced_channels) for _ in range(self.v_num)])
        self.bn_policy = nn.BatchNorm2d(reduced_channels)
        self.block_output_size_value = flatten_size
        self.block_output_size_policy = flatten_size
        self.fc_values = nn.ModuleList([mlp(self.block_output_size_value, fc_layers, value_output_size,
                            init_zero=False if is_continuous else init_zero) for _ in range(self.v_num)])
        self.fc_policy = mlp(self.block_output_size_policy, fc_layers if not is_continuous else [64],
                             policy_output_size, init_zero=init_zero)

        self.is_continuous = is_continuous
        self.value_policy_detach = kwargs.get('value_policy_detach')

        self.init_std = 1.0
        self.min_std = 0.1

    def forward(self, x, task_embedding=None):
        if self.multi_task:
            x = torch.cat([x, task_embedding], dim=1)
        if self.value_policy_detach:
            x = x.detach()  # try decoupled training

        for block in self.resblocks:
            x = block(x)

        values = []
        for i in range(self.v_num):
            value = self.conv1x1_values[i](x)
            value = self.bn_values[i](value)
            value = nn.functional.relu(value)
            value = value.reshape(-1, self.block_output_size_value)
            value = self.fc_values[i](value)
            values.append(value)

        policy = self.conv1x1_policy(x)
        policy = self.bn_policy(policy)
        policy = nn.functional.relu(policy)
        policy = policy.reshape(-1, self.block_output_size_policy)
        policy = self.fc_policy(policy)

        if self.is_continuous:
            action_space_size = policy.shape[-1] // 2
            policy[:, :action_space_size] = 5 * torch.tanh(policy[:, :action_space_size] / 5)  # soft clamp mu
            policy[:, action_space_size:] = (torch.nn.functional.softplus(policy[:, action_space_size:] + self.init_std) + self.min_std)#.clip(0, 5)  # same as Dreamer-v3

        return torch.stack(values), policy

class SupportNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, reduced_channels, flatten_size, fc_layers, output_support_size, init_zero, multi_task=False, task_embed_channels=16):
        super().__init__()
        self.flatten_size = flatten_size
        self.multi_task = multi_task
        if not multi_task:
            task_embed_channels = 0
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels + task_embed_channels, num_channels + task_embed_channels) for _ in
             range(num_blocks)]
        )
        self.conv1x1 = nn.Conv2d(num_channels + task_embed_channels, reduced_channels, 1)
        self.bn = nn.BatchNorm2d(reduced_channels)
        self.fc = mlp(flatten_size, fc_layers, output_support_size, init_zero=init_zero)

    def forward(self, x, task_embedding=None):
        if self.multi_task:
            x = torch.cat([x, task_embedding], dim=1)

        for block in self.resblocks:
            x = block(x)

        x = self.conv1x1(x)
        x = self.bn(x)
        x = nn.functional.relu(x)
        # x = nn.functional.avg_pool2d(x, 2)
        x = x.reshape(-1, self.flatten_size)
        x = self.fc(x)
        return x

class SupportNetworkParallel(nn.Module):
    def __init__(self, num_blocks, num_channels, action_space_size,
                 reduced_channels, flatten_size, fc_layers, output_support_size, init_zero, is_continuous=False,
                 action_embedding=False, action_embedding_dim=32,
                 multi_task=False, task_embed_channels=16):
        """
        Dynamics network
        :param num_blocks: int, number of res blocks
        :param num_channels: int, channels of hidden states
        :param action_space_size: int, action space size
        """
        super().__init__()
        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        self.num_channels = num_channels
        self.action_space_size = action_space_size
        self.multi_task = multi_task
        self.flatten_size = flatten_size
        if not multi_task:
            task_embed_channels = 0

        action_space_dim = action_space_size if is_continuous else 1

        if action_embedding:
            self.conv1x1 = nn.Conv2d(action_space_dim + task_embed_channels, self.action_embedding_dim, 1)
            self.ln = nn.LayerNorm([action_embedding_dim, 6, 6])
            self.conv = conv3x3(num_channels + task_embed_channels + self.action_embedding_dim, num_channels)
        else:
            self.conv = conv3x3(num_channels + task_embed_channels + action_space_dim, num_channels)

        self.bn = nn.BatchNorm2d(num_channels)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

        self.conv1x1_rew = nn.Conv2d(num_channels, reduced_channels, 1)
        self.bn_rew = nn.BatchNorm2d(reduced_channels)
        self.fc_rew = mlp(flatten_size, fc_layers, output_support_size, init_zero=init_zero)

    def forward(self, state, action, task_embedding=None):
        # encode action
        if not self.is_continuous:
            action_place = torch.ones((
                state.shape[0],
                1,
                state.shape[2],
                state.shape[3],
            )).cuda().float()

            action_place = (
                    action[:, :, None, None] * action_place / self.action_space_size
            )
        else:
            action_place = action.reshape(*action.shape, 1, 1).repeat(1, 1, state.shape[-2], state.shape[-1])

        if self.action_embedding:
            if self.multi_task:
                action_input = torch.cat([action_place, task_embedding], dim=1)
            else:
                action_input = action_place
            action_place = self.conv1x1(action_input)
            action_place = self.ln(action_place)
            action_place = nn.functional.relu(action_place)

        if self.multi_task:
            x = torch.cat([state, task_embedding], dim=1)
        else:
            x = state
        x = torch.cat((x, action_place), dim=1)
        x = self.conv(x)
        x = self.bn(x)

        x = x + state
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)

        x = self.conv1x1_rew(x)
        x = self.bn_rew(x)
        x = nn.functional.relu(x)
        x = x.reshape(-1, self.flatten_size)
        x = self.fc_rew(x)
        return x


class SupportLSTMNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, reduced_channels, flatten_size, fc_layers, output_support_size, lstm_hidden_size, init_zero, multi_task=False, task_embed_channels=16):
        super().__init__()
        self.flatten_size = flatten_size
        self.multi_task = multi_task
        if not multi_task:
            task_embed_channels = 0

        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels + task_embed_channels, num_channels + task_embed_channels) for _ in
             range(num_blocks)]
        )

        self.conv1x1_reward = nn.Conv2d(num_channels + task_embed_channels, reduced_channels, 1)
        self.bn_reward = nn.BatchNorm2d(reduced_channels)
        self.lstm = nn.LSTM(input_size=flatten_size, hidden_size=lstm_hidden_size)
        self.bn_reward_sum = nn.BatchNorm1d(lstm_hidden_size)
        self.fc = mlp(lstm_hidden_size, fc_layers, output_support_size, init_zero=init_zero)

    def forward(self, x, hidden, task_embedding=None):
        if self.multi_task:
            x = torch.cat([x, task_embedding], dim=1)

        for block in self.resblocks:
            x = block(x)

        x = self.conv1x1_reward(x)
        x = self.bn_reward(x)
        x = nn.functional.relu(x)
        # x = nn.functional.avg_pool2d(x, 2)
        x = x.reshape(-1, self.flatten_size).unsqueeze(0)
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(0)
        x = self.bn_reward_sum(x)
        x = nn.functional.relu(x)
        x = self.fc(x)
        return x, hidden


class SupportLSTMNetworkParallel(nn.Module):
    def __init__(self, num_blocks, num_channels, action_space_size,
                 reduced_channels, flatten_size, fc_layers, output_support_size, lstm_hidden_size, init_zero, is_continuous=False,
                 action_embedding=False, action_embedding_dim=32,
                 multi_task=False, task_embed_channels=16):
        """
        Dynamics network
        :param num_blocks: int, number of res blocks
        :param num_channels: int, channels of hidden states
        :param action_space_size: int, action space size
        """
        super().__init__()
        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        self.num_channels = num_channels
        self.action_space_size = action_space_size
        self.multi_task = multi_task
        self.flatten_size = flatten_size
        if not multi_task:
            task_embed_channels = 0

        action_space_dim = action_space_size if is_continuous else 1

        if action_embedding:
            self.conv1x1 = nn.Conv2d(action_space_dim + task_embed_channels, self.action_embedding_dim, 1)
            self.ln = nn.LayerNorm([action_embedding_dim, 6, 6])
            self.conv = conv3x3(num_channels + task_embed_channels + self.action_embedding_dim, num_channels)
        else:
            self.conv = conv3x3(num_channels + task_embed_channels + action_space_dim, num_channels)

        self.bn = nn.BatchNorm2d(num_channels)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

        self.conv1x1_reward = nn.Conv2d(num_channels, reduced_channels, 1)
        self.bn_reward = nn.BatchNorm2d(reduced_channels)
        self.lstm = nn.LSTM(input_size=flatten_size, hidden_size=lstm_hidden_size)
        self.bn_reward_sum = nn.BatchNorm1d(lstm_hidden_size)
        self.fc = mlp(lstm_hidden_size, fc_layers, output_support_size, init_zero=init_zero)

    def forward(self, state, action, hidden, task_embedding=None):
        # encode action
        if not self.is_continuous:
            action_place = torch.ones((
                state.shape[0],
                1,
                state.shape[2],
                state.shape[3],
            )).cuda().float()

            action_place = (
                    action[:, :, None, None] * action_place / self.action_space_size
            )
        else:
            action_place = action.reshape(*action.shape, 1, 1).repeat(1, 1, state.shape[-2], state.shape[-1])

        if self.action_embedding:
            if self.multi_task:
                action_input = torch.cat([action_place, task_embedding], dim=1)
            else:
                action_input = action_place
            action_place = self.conv1x1(action_input)
            action_place = self.ln(action_place)
            action_place = nn.functional.relu(action_place)

        if self.multi_task:
            x = torch.cat([state, task_embedding], dim=1)
        else:
            x = state
        x = torch.cat((x, action_place), dim=1)
        x = self.conv(x)
        x = self.bn(x)

        x = x + state
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)

        x = self.conv1x1_reward(x)
        x = self.bn_reward(x)
        x = nn.functional.relu(x)
        x = x.reshape(-1, self.flatten_size).unsqueeze(0)
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(0)
        x = self.bn_reward_sum(x)
        x = nn.functional.relu(x)
        x = self.fc(x)
        return x, hidden


class ProjectionNetwork(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super().__init__()

        self.input_dim = input_dim
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.layer(x)


class ProjectionHeadNetwork(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        return self.layer(x)
