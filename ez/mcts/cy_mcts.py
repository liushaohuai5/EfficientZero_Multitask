import copy

import torch
# import torchrl
import numpy as np
import math
import random
from ez.mcts.ctree_v2 import cytree as tree2
from torch.cuda.amp import autocast as autocast
from ez.utils.format import log_std_transform, gaussian_logprob, squash
from ez.utils.distribution import SquashedNormal, TruncatedNormal, ContDist


class Gumbel_MCTS(object):

    def __init__(self, config):
        self.config = config
        self.value_prefix = self.config.model.value_prefix
        self.num_simulations = self.config.mcts.num_simulations
        self.num_top_actions = self.config.mcts.num_top_actions
        self.c_visit = self.config.mcts.c_visit
        self.c_scale = self.config.mcts.c_scale
        self.discount = self.config.rl.discount
        self.value_minmax_delta = self.config.mcts.value_minmax_delta
        self.lstm_hidden_size = self.config.model.lstm_hidden_size
        self.action_space_size = self.config.env.action_space_size
        self.policy_action_num = self.config.model.policy_action_num if not self.config.env.env == 'Atari' else 1
        self.random_action_num = self.config.model.random_action_num if not self.config.env.env == 'Atari' else 1
        try:
            self.policy_distribution = self.config.model.policy_distribution
        except:
            pass
        self.max_horizon = self.config.mcts.max_horizon
        self.gumbel_noise = None

    def update_statistics(self, **kwargs):
        if kwargs.get('prediction'):
            # prediction for next states, rewards, values, logits
            model = kwargs.get('model')
            current_states = kwargs.get('states')
            last_actions = kwargs.get('actions')
            reward_hidden = kwargs.get('reward_hidden')
            task_idxs = kwargs.get('task_idxs')

            with torch.no_grad():
                with autocast():
                    next_states, next_value_prefixes, next_values, next_logits, reward_hidden, variances = \
                        model.recurrent_inference(current_states, last_actions, reward_hidden, task_idxs=task_idxs)

            # process outputs
            next_values = next_values.detach().cpu().numpy().flatten()
            next_value_prefixes = next_value_prefixes.detach().cpu().numpy().flatten()
            return next_states, next_value_prefixes, next_values, next_logits, reward_hidden, variances
        else:
            # env simulation for next states
            env = kwargs.get('env')
            current_states = kwargs.get('states')
            last_actions = kwargs.get('actions')
            states = env.step(last_actions)
            raise NotImplementedError()


    def pi_sample(self, policy, add_noise=True):
        n_policy = self.policy_action_num
        n_random = self.random_action_num
        ratio = n_random / (n_policy + n_random)
        mean, log_std = policy.chunk(2, dim=-1)
        log_std = log_std_transform(log_std, low=-10, dif=12)
        eps = torch.randn_like(mean)
        log_prob = gaussian_logprob(eps, log_std)

        std = log_std.exp()
        if add_noise and std.size(1) > 1:
            std[:, -int(ratio*std.size(1)):] *= 3
        action = mean + eps * std
        mean, action, log_prob = squash(mean, action, log_prob)
        return action


    def sample_actions(self, policy, add_noise=True, temperature=1.0, sample_nums=None):
        n_policy = self.policy_action_num
        n_random = self.random_action_num
        if sample_nums:
            assert sample_nums % 2 == 0, 'sample_nums must be even'
            n_policy, n_random = sample_nums // 2, sample_nums // 2
        std_magnification = self.config.mcts.std_magnification

        if self.config.model.policy_distribution == 'tdmpc2':
            policy_expanded = policy.unsqueeze(1).repeat(1, n_policy+n_random, 1)
            sampled_actions = self.pi_sample(policy_expanded)

            policy_actions = sampled_actions[:, :n_policy]
            random_actions = sampled_actions[:, -n_random:]

            all_actions = torch.cat((policy_actions, random_actions), dim=1)
            all_actions = all_actions.clip(-0.999, 0.999)
            sample_logits = np.zeros((all_actions.shape[0], all_actions.shape[1])).tolist()
            sample_ind = np.zeros((all_actions.shape[0], all_actions.shape[1], all_actions.shape[2]))

        elif self.config.model.policy_distribution == 'squashed_gaussian':
            Dist = SquashedNormal
            mean, std = policy.chunk(2, dim=-1)
            distr = Dist(mean, std)
            sampled_actions = distr.sample(torch.Size([n_policy + n_random]))

            policy_actions = sampled_actions[:n_policy]
            random_actions = sampled_actions[-n_random:]

            if add_noise:
                random_distr = Dist(mean, std_magnification * std)  # more flatten gaussian policy
                random_actions = random_distr.sample(torch.Size([n_random]))

            all_actions = torch.cat((policy_actions, random_actions), dim=0)
            all_actions = all_actions.clip(-0.999, 0.999)
            all_actions = all_actions.permute(1, 0, 2)
            sample_logits = np.zeros((all_actions.shape[0], all_actions.shape[1])).tolist()
            sample_ind = np.zeros((all_actions.shape[0], all_actions.shape[1], all_actions.shape[2]))

        return all_actions, sample_logits, sample_ind

    def atanh(self, x):
        return 0.5 * (np.log1p(x) - np.log1p(-x))

    @torch.no_grad()
    def run_multi_discrete(
        self, model, batch_size,
        hidden_state_roots, root_values,
        root_policy_logits, temperature=1.0,
        use_gumbel_noise=True,
        task_idxs=None,
    ):

        model.eval()

        reward_sum_pool = [0. for _ in range(batch_size)]

        roots = tree2.Roots(
            batch_size, self.action_space_size,
            self.num_simulations
        )
        root_policy_logits = root_policy_logits.detach().cpu().numpy()

        roots.prepare(
            reward_sum_pool, root_policy_logits.tolist(),
            self.num_top_actions, self.num_simulations,
            root_values.tolist(), self.action_space_size
        )

        reward_hidden_roots = (
            torch.from_numpy(np.zeros((1, batch_size, self.lstm_hidden_size))).float().cuda(),
            torch.from_numpy(np.zeros((1, batch_size, self.lstm_hidden_size))).float().cuda()
        )

        gumbels = np.random.gumbel(
            0, 1, (batch_size, self.action_space_size)
        )# * temperature
        if not use_gumbel_noise:
            gumbels = np.zeros_like(gumbels)
        gumbels = gumbels.tolist()

        num = roots.num
        c_visit, c_scale, discount = self.c_visit, self.c_scale, self.discount
        hidden_state_pool = [hidden_state_roots]
        # 1 x batch x 64
        reward_hidden_c_pool = [reward_hidden_roots[0]]
        reward_hidden_h_pool = [reward_hidden_roots[1]]
        hidden_state_index_x = 0
        min_max_stats_lst = tree2.MinMaxStatsList(num)
        min_max_stats_lst.set_delta(self.value_minmax_delta)
        horizons = self.config.model.lstm_horizon_len

        variances_sum = np.zeros((batch_size, 1))
        for index_simulation in range(self.num_simulations):
            hidden_states = []
            hidden_states_c_reward = []
            hidden_states_h_reward = []

            results = tree2.ResultsWrapper(num)
            hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions, _, _, _ = \
                tree2.multi_traverse(
                    roots, c_visit, c_scale, discount,
                    min_max_stats_lst, results,
                    index_simulation, gumbels,
                    int(False)
                )
            search_lens = results.get_search_len()

            for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                hidden_states.append(hidden_state_pool[ix][iy].unsqueeze(0))
                if self.value_prefix:
                    hidden_states_c_reward.append(reward_hidden_c_pool[ix][0][iy].unsqueeze(0))
                    hidden_states_h_reward.append(reward_hidden_h_pool[ix][0][iy].unsqueeze(0))

            hidden_states = torch.cat(hidden_states, dim=0)
            if self.value_prefix:
                hidden_states_c_reward = torch.cat(hidden_states_c_reward).unsqueeze(0)
                hidden_states_h_reward = torch.cat(hidden_states_h_reward).unsqueeze(0)

            last_actions = torch.from_numpy(
                np.asarray(last_actions)
            ).to('cuda').unsqueeze(1).long()

            hidden_state_nodes, reward_sum_pool, value_pool, policy_logits_pool, reward_hidden_nodes, variances = \
                self.update_statistics(
                    prediction=True,  # use model prediction instead of env simulation
                    model=model,  # model
                    states=hidden_states,  # current states
                    actions=last_actions,  # last actions
                    reward_hidden=(hidden_states_c_reward, hidden_states_h_reward),  # reward hidden
                    task_idxs=torch.from_numpy(task_idxs).long().cuda(),
                )

            variances_sum += variances.detach().cpu().numpy()
            reward_sum_pool = reward_sum_pool.tolist()
            value_pool = value_pool.tolist()
            policy_logits_pool = policy_logits_pool.detach().cpu().numpy().tolist()

            hidden_state_pool.append(hidden_state_nodes)
            # reset 0
            if self.value_prefix:
                if horizons > 0:
                    reset_idx = (np.array(search_lens) % horizons == 0)
                    assert len(reset_idx) == num
                    reward_hidden_nodes[0][:, reset_idx, :] = 0
                    reward_hidden_nodes[1][:, reset_idx, :] = 0
                    is_reset_lst = reset_idx.astype(np.int32).tolist()
                else:
                    is_reset_lst = [0 for _ in range(num)]
            else:
                is_reset_lst = [1 for _ in range(num)]

            if self.value_prefix:
                reward_hidden_c_pool.append(reward_hidden_nodes[0])
                reward_hidden_h_pool.append(reward_hidden_nodes[1])
            hidden_state_index_x += 1

            tree2.multi_back_propagate(
                hidden_state_index_x, discount,
                reward_sum_pool, value_pool, policy_logits_pool,
                min_max_stats_lst, results, is_reset_lst,
                index_simulation, gumbels, c_visit, c_scale, self.num_simulations, self.action_space_size
            )

        root_values = np.asarray(roots.get_values())
        pi_primes = np.asarray(roots.get_pi_primes(
            min_max_stats_lst, c_visit, c_scale, discount
        ))
        best_actions = np.asarray(roots.get_actions(
            min_max_stats_lst, c_visit, c_scale, gumbels, discount
        ))
        root_sampled_actions = np.expand_dims(
            np.arange(self.action_space_size), axis=0
        ).repeat(batch_size, axis=0)

        return root_values, pi_primes, best_actions, \
               min_max_stats_lst.get_min_max(), root_sampled_actions, variances_sum

    def run_multi_continuous(
            self, model, batch_size,
            hidden_state_roots, root_values,
            root_policy_logits, temperature=1.0, add_noise=True, use_gumbel_noise=False, task_idxs=None
    ):
        with (torch.no_grad()):
            model.eval()

            reward_sum_pool = [0. for _ in range(batch_size)]
            reward_hidden_roots = (
                torch.from_numpy(np.zeros((1, batch_size, self.lstm_hidden_size))).float().cuda(),
                torch.from_numpy(np.zeros((1, batch_size, self.lstm_hidden_size))).float().cuda()
            )

            root_sampled_actions, uniform_policy, sample_indices = self.sample_actions(root_policy_logits, add_noise, temperature)
            sampled_action_num = root_sampled_actions.shape[1]

            roots = tree2.Roots(
                batch_size, sampled_action_num, self.num_simulations
            )

            leaf_num = 2

            roots.prepare(
                reward_sum_pool,
                uniform_policy,
                self.num_top_actions,
                self.num_simulations,
                root_values.tolist(),
                leaf_num
            )

            gumbels = np.random.gumbel(
                0, 1, (batch_size, sampled_action_num)
            ) * temperature
            if not use_gumbel_noise:
                gumbels = np.zeros_like(gumbels)
            gumbels = gumbels.tolist()

            num = roots.num
            c_visit, c_scale, discount = self.c_visit, self.c_scale, self.discount
            hidden_state_pool = [hidden_state_roots]
            # 1 x batch x 64
            reward_hidden_c_pool = [reward_hidden_roots[0]]
            reward_hidden_h_pool = [reward_hidden_roots[1]]
            hidden_state_index_x = 0
            min_max_stats_lst = tree2.MinMaxStatsList(num)
            min_max_stats_lst.set_delta(self.value_minmax_delta)
            horizons = self.config.model.lstm_horizon_len

            actions_pool = [root_sampled_actions]
            variances_sum = np.zeros((batch_size, 1))
            for index_simulation in range(self.num_simulations):
                hidden_states = []
                hidden_states_c_reward = []
                hidden_states_h_reward = []

                results = tree2.ResultsWrapper(num)
                hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions, first_actions, search_lens, _ = \
                    tree2.multi_traverse(roots, c_visit, c_scale, discount, min_max_stats_lst,
                                         results, index_simulation, gumbels, int(self.config.model.dynamic_type == 'Transformer'))
                search_lens = results.get_search_len()

                ptr = 0
                selected_actions = []
                for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                    # ix - simulation idx, iy - minibatch idx
                    hidden_states.append(hidden_state_pool[ix][iy].unsqueeze(0))
                    if self.value_prefix:
                        hidden_states_c_reward.append(reward_hidden_c_pool[ix][0][iy].unsqueeze(0))
                        hidden_states_h_reward.append(reward_hidden_h_pool[ix][0][iy].unsqueeze(0))
                    selected_actions.append(
                        actions_pool[ix][iy][last_actions[ptr]].unsqueeze(0)
                    )
                    ptr += 1

                hidden_states = torch.cat(hidden_states, dim=0).float()
                if self.value_prefix:
                    hidden_states_c_reward = torch.cat(hidden_states_c_reward, dim=0).unsqueeze(0)
                    hidden_states_h_reward = torch.cat(hidden_states_h_reward, dim=0).unsqueeze(0)

                selected_actions = torch.cat(selected_actions, dim=0).float()
                hidden_state_nodes, reward_sum_pool, value_pool, policy_logits_pool, reward_hidden_nodes, variances = self.update_statistics(
                    prediction=True,  # use model prediction instead of env simulation
                    model=model,  # model
                    states=hidden_states,  # current states
                    actions=selected_actions,  # last actions
                    reward_hidden=(hidden_states_c_reward, hidden_states_h_reward),  # reward hidden
                    task_idxs=torch.from_numpy(task_idxs).long().cuda(),
                )
                variances_sum += variances.detach().cpu().numpy()

                leaf_sampled_actions, uniform_policy_non_root, _ = self.sample_actions(policy_logits_pool, add_noise=False, sample_nums=leaf_num)
                actions_pool.append(leaf_sampled_actions)
                reward_sum_pool = reward_sum_pool.tolist()
                value_pool = value_pool.tolist()

                hidden_state_pool.append(hidden_state_nodes)
                # reset 0
                if self.value_prefix:
                    if horizons > 0:
                        reset_idx = (np.array(search_lens) % horizons == 0)
                        assert len(reset_idx) == num
                        reward_hidden_nodes[0][:, reset_idx, :] = 0
                        reward_hidden_nodes[1][:, reset_idx, :] = 0
                        is_reset_lst = reset_idx.astype(np.int32).tolist()
                    else:
                        is_reset_lst = [0 for _ in range(num)]
                else:
                    is_reset_lst = [1 for _ in range(num)]      # TODO: this is a huge bug, previous 0.

                if self.value_prefix:
                    reward_hidden_c_pool.append(reward_hidden_nodes[0])
                    reward_hidden_h_pool.append(reward_hidden_nodes[1])
                assert hidden_state_index_x == index_simulation, 'index_simulation != hidden_state_index_x'
                hidden_state_index_x += 1

                tree2.multi_back_propagate(
                    hidden_state_index_x, discount,
                    reward_sum_pool, value_pool,
                    uniform_policy_non_root,
                    min_max_stats_lst, results, is_reset_lst,
                    index_simulation, gumbels, c_visit, c_scale, self.num_simulations, leaf_num
                )

        best_action_idxs = roots.get_actions(min_max_stats_lst, c_visit, c_scale, gumbels, discount)
        root_sampled_actions = root_sampled_actions.detach().cpu().numpy()
        final_selected_actions = np.asarray(
            [root_sampled_actions[i, best_a] for i, best_a in enumerate(best_action_idxs)]
        )
        pi_primes = np.asarray(roots.get_pi_primes(min_max_stats_lst, c_visit, c_scale, discount))

        return np.asarray(roots.get_values()), \
               pi_primes, \
               np.asarray(final_selected_actions), min_max_stats_lst.get_min_max(), \
               np.asarray(root_sampled_actions), np.asarray(best_action_idxs), variances_sum