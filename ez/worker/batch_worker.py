import os
import time
# import SMOS
import ray
import torch
import copy
import gym
import imageio
from PIL import Image, ImageDraw
import numpy as np

from torch.cuda.amp import autocast as autocast

from .base import Worker
from ez.utils.format import formalize_obs_lst, LinearSchedule, prepare_obs_lst, set_seed
from ez.data.trajectory import GameTrajectory
from ez.mcts.cy_mcts import Gumbel_MCTS

@ray.remote(num_gpus=0.1)
class BatchWorker(Worker):
    def __init__(self, rank, agent, replay_buffer, storage, batch_storage, config, seed=None):
        super().__init__(rank, agent, replay_buffer, storage, config)

        self.model_update_interval = config.train.reanalyze_update_interval
        self.batch_storage = batch_storage

        self.beta_schedule = LinearSchedule(self.total_steps, initial_p=config.priority.priority_prob_beta, final_p=1.0)
        self.total_transitions = self.config.data.total_transitions
        self.auto_td_steps = self.config.rl.auto_td_steps
        self.td_steps = self.config.rl.td_steps
        self.unroll_steps = self.config.rl.unroll_steps
        self.n_stack = self.config.env.n_stack
        self.discount = self.config.rl.discount
        self.value_support = self.config.model.value_support
        self.action_space_size = self.config.env.action_space_size
        self.batch_size = self.config.train.batch_size
        self.PER_alpha = self.config.priority.priority_prob_alpha
        self.env = self.config.env.env
        self.continuous_action = config.env.continuous_action
        self.image_based = self.config.env.image_based
        self.reanalyze_ratio = self.config.train.reanalyze_ratio
        self.value_target = self.config.train.value_target
        self.value_target_type = self.config.model.value_target
        self.GAE_max_steps = self.config.model.GAE_max_steps
        self.episodic = self.config.env.episodic
        self.value_prefix = self.config.model.value_prefix
        self.lstm_horizon_len = self.config.model.lstm_horizon_len
        self.training_steps = self.config.train.training_steps
        self.td_lambda = self.config.rl.td_lambda
        self.search_type = self.config.mcts.search_type
        self.gray_scale = self.config.env.gray_scale
        self.obs_shape = self.config.env.obs_shape
        self.trajectory_size = self.config.data.trajectory_size
        self.mixed_value_threshold = self.config.train.mixed_value_threshold
        self.lstm_hidden_size = self.config.model.lstm_hidden_size
        self.cnt = 0

        if seed is not None:
            self.seed = seed
            set_seed(seed)

    def concat_trajs(self, items):
        obs_lsts, reward_lsts, policy_lsts, policy_prior_lsts, action_lsts, pred_value_lsts, search_value_lsts, bootstrapped_value_lsts, task_idx_lsts = items
        traj_lst = []
        for obs_lst, reward_lst, policy_lst, policy_prior_lst, action_lst, pred_value_lst, search_value_lst, bootstrapped_value_lst, task_idx_lst in \
                zip(obs_lsts, reward_lsts, policy_lsts, policy_prior_lsts, action_lsts, pred_value_lsts, search_value_lsts, bootstrapped_value_lsts, task_idx_lsts):
            traj = GameTrajectory(
                n_stack=self.n_stack, discount=self.discount, gray_scale=self.gray_scale, unroll_steps=self.unroll_steps,
                td_steps=self.td_steps, td_lambda=self.td_lambda, obs_shape=self.obs_shape, max_size=self.trajectory_size,
                image_based=self.image_based, episodic=self.episodic, GAE_max_steps=self.GAE_max_steps
            )
            traj.obs_lst = obs_lst
            traj.reward_lst = reward_lst
            traj.policy_lst = policy_lst
            traj.policy_prior_lst = policy_prior_lst
            traj.action_lst = action_lst
            traj.pred_value_lst = pred_value_lst
            traj.search_value_lst = search_value_lst
            traj.bootstrapped_value_lst = bootstrapped_value_lst
            traj.task_idx_lst = task_idx_lst
            traj_lst.append(traj)
        return traj_lst

    def run(self):
        trained_steps = 0

        # create the model for self-play data collection
        self.model = self.agent.build_model()
        self.latest_model = self.agent.build_model()
        if self.config.eval.analysis_value:
            weights = torch.load(self.config.eval.model_path)
            self.model.load_state_dict(weights)
            print('analysis begin')
        self.model.cuda()
        self.latest_model.cuda()
        if int(torch.__version__[0]) == 2:
            self.model = torch.compile(self.model)
            self.latest_model = torch.compile(self.latest_model)
        self.model.eval()
        self.latest_model.eval()
        self.resume_model()

        # wait for starting to train
        while not ray.get(self.storage.get_start_signal.remote()):
            time.sleep(0.5)

        # begin to make batch
        prev_trained_steps = -10
        while not self.is_finished(trained_steps):
            trained_steps = ray.get(self.storage.get_counter.remote())
            if self.config.ray.single_process:
                if trained_steps <= prev_trained_steps:
                    time.sleep(0.1)
                    continue
                prev_trained_steps = trained_steps
                print(f'reanalyze[{self.rank}] makes batch at step {trained_steps}')
            # get the fresh model weights
            self.get_recent_model(trained_steps, 'reanalyze')
            self.get_latest_model(trained_steps, 'latest')

            ray_time = self.make_batch(trained_steps, self.cnt)
            self.cnt += 1

    def make_batch(self, trained_steps, cnt, real_time=False):
        beta = self.beta_schedule.value(trained_steps)
        batch_size = self.batch_size

        # obtain the batch context from replay buffer
        x = time.time()
        batch_context = ray.get(
            self.replay_buffer.prepare_batch_context.remote(batch_size=batch_size,
                                                            alpha=self.PER_alpha,
                                                            beta=beta,
                                                            rank=self.rank,
                                                            cnt=cnt)
        )
        batch_context, validation_flag = batch_context

        ray_time = time.time() - x
        traj_lst, transition_pos_lst, indices_lst, weights_lst, make_time_lst, transition_num, prior_lst = batch_context
        traj_lst = self.concat_trajs(traj_lst)

        # part of policy will be reanalyzed
        reanalyze_batch_size = batch_size if self.continuous_action else int(batch_size * self.config.train.reanalyze_ratio)
        assert 0 <= reanalyze_batch_size <= batch_size

        # ==============================================================================================================
        # make inputs
        # ==============================================================================================================
        collected_transitions = ray.get(self.replay_buffer.get_transition_num.remote())
        # make observations, actions and masks (if unrolled steps are out of trajectory)
        obs_lst, action_lst, policy_prior_lst, mask_lst, task_idx_lst = [], [], [], [], []
        top_new_masks = []
        # prepare the inputs of a batch
        for i in range(batch_size):
            traj = traj_lst[i]
            state_index = transition_pos_lst[i]
            sample_idx = indices_lst[i]
            task_idx = traj.task_idx_lst[state_index] if self.config.data.ind_exp_rp else 0

            top_new_masks.append(int(sample_idx > collected_transitions[task_idx] - self.mixed_value_threshold))

            if self.continuous_action:
                _actions = traj.action_lst[state_index:state_index + self.unroll_steps]
                _unroll_actions = traj.action_lst[state_index + 1:state_index + 1 + self.unroll_steps]
                # _unroll_actions = traj.action_lst[state_index:state_index + self.unroll_steps]
                _mask = [1. for _ in range(_unroll_actions.shape[0])]
                _mask += [0. for _ in range(self.unroll_steps - len(_mask))]
                _rand_actions = np.zeros((self.unroll_steps - _actions.shape[0], self.action_space_size))
                _actions = np.concatenate((_actions, _rand_actions), axis=0)
            else:
                _actions = traj.action_lst[state_index:state_index + self.unroll_steps].tolist()
                _mask = [1. for _ in range(len(_actions))]
                _mask += [0. for _ in range(self.unroll_steps - len(_mask))]
                _actions += [np.random.randint(0, self.action_space_size) for _ in range(self.unroll_steps - len(_actions))]

            # obtain the input observations
            obs_lst.append(traj.get_index_stacked_obs(state_index, padding=True))
            action_lst.append(_actions)
            policy_prior_lst.append(traj.policy_prior_lst[state_index:state_index + 1 + self.unroll_steps])
            mask_lst.append(_mask)

            task_idx_lst.append(traj.task_idx_lst[state_index])

        task_idx_lst = np.repeat(np.array(task_idx_lst)[:, None], self.unroll_steps+1, axis=1)

        obs_lst = prepare_obs_lst(obs_lst, self.image_based)
        inputs_batch = [obs_lst, action_lst, policy_prior_lst, mask_lst, indices_lst, weights_lst, make_time_lst, prior_lst, task_idx_lst]
        for i in range(len(inputs_batch)):
            inputs_batch[i] = np.asarray(inputs_batch[i])

        # ==============================================================================================================
        # make targets
        # ==============================================================================================================

        if self.value_target in ['sarsa', 'mixed', 'max']:
            prepare_func = self.prepare_reward_value
        elif self.value_target == 'search':
            prepare_func = self.prepare_reward
        else:
            raise NotImplementedError


        # obtain the value prefix (reward), and the value
        task_idxs = task_idx_lst.reshape(-1)
        batch_value_prefixes, batch_values, td_steps, batch_variances, pre_calc = \
            prepare_func(traj_lst, transition_pos_lst, indices_lst, collected_transitions, trained_steps, task_idxs)

        # obtain the re policy
        if reanalyze_batch_size > 0:
            batch_policies_re, sampled_actions, best_actions, reanalyzed_values, reanalyzed_variances, pre_lst, policy_masks = \
                self.prepare_policy_reanalyze(
                trained_steps, traj_lst[:reanalyze_batch_size], transition_pos_lst[:reanalyze_batch_size],
                indices_lst[:reanalyze_batch_size],
                state_lst=pre_calc[0], value_lst=pre_calc[1], policy_lst=pre_calc[2], policy_mask=pre_calc[3],
                task_idxs=task_idxs
            )
        else:
            batch_policies_re = []

        batch_variances += np.asarray(reanalyzed_variances).squeeze()

        self.storage.update_variance_minmax.remote(batch_variances.flatten(), task_idxs)
        variance_max, variance_min = ray.get(self.storage.get_variance_minmax.remote())
        variance_max = variance_max[task_idxs].reshape(batch_size, -1)
        variance_min = variance_min[task_idxs].reshape(batch_size, -1)
        if self.rank == 0:
            new_log_index = trained_steps // 10
            if new_log_index > self.last_log_index:
                self.last_log_index = new_log_index
                self.storage.add_log_distribution.remote({
                    'batch_worker/value_variance': batch_variances.flatten(),
                })
        # TODO: check if periodic oscillation disappear after removing variance-controlled mixed value targets.
        if self.config.train.mixed_type == 'variance':
            top_new_masks = ((batch_variances - variance_min) / (variance_max - variance_min)).clip(0, 1)
        else:
            top_new_masks = np.expand_dims(np.asarray(top_new_masks), axis=1).repeat(self.unroll_steps+1, axis=1)

        # concat target policy
        batch_policies = batch_policies_re
        if self.continuous_action:
            batch_best_actions = best_actions.reshape(batch_size, self.unroll_steps + 1,
                                                      self.action_space_size)
        else:
            batch_best_actions = np.asarray(best_actions).reshape(batch_size,
                                                                  self.unroll_steps + 1)

        # target value prefix (reward), value, policy
        if self.env not in ['DMC', 'Gym', 'Humanoid_Bench', 'ManiSkill']:
            batch_actions = np.ones_like(batch_policies)
        else:
            batch_actions = sampled_actions.reshape(
                batch_size, self.unroll_steps + 1, -1, self.action_space_size
            )
        targets_batch = [batch_value_prefixes, batch_values, batch_actions, batch_policies, batch_best_actions, top_new_masks, policy_masks, reanalyzed_values]

        for i in range(len(targets_batch)):
            targets_batch[i] = np.asarray(targets_batch[i])

        # ==============================================================================================================
        # push batch into batch queue
        # ==============================================================================================================
        # full batch data: [obs_lst, other stuffs, target stuffs]
        batch = [inputs_batch, targets_batch]

        # log
        self.storage.add_log_scalar.remote({
            'batch_worker/td_step': np.mean(td_steps)
        })

        if real_time:
            return batch
        else:
            # push into batch storage
            self.batch_storage.push(batch)

        return ray_time


    def prepare_reward(self, traj_lst, transition_pos_lst, indices_lst, collected_transitions, trained_steps, task_idxs=None):
        # value prefix (or reward), value
        batch_value_prefixes = []

        # v_{t} = r + ... + gamma ^ k * v_{t+k}
        value_index = 0
        # top_value_masks = []
        for traj, state_index, idx in zip(traj_lst, transition_pos_lst, indices_lst):
            traj_len = len(traj.reward_lst)
            # traj_len = len(traj)
            target_value_prefixs = []

            horizon_id = 0
            value_prefix = 0.0
            # top_value_masks.append(int(idx > collected_transitions - 4e4))
            for current_index in range(state_index, state_index + self.unroll_steps + 1):

                # reset every lstm_horizon_len
                if horizon_id % self.lstm_horizon_len == 0 and self.value_prefix:
                    value_prefix = 0.0
                horizon_id += 1

                if current_index < traj_len:
                    # Since the horizon is small and the discount is close to 1.
                    # Compute the reward sum to approximate the value prefix for simplification
                    if self.value_prefix:
                        value_prefix += traj.reward_lst[current_index]
                    else:
                        value_prefix = traj.reward_lst[current_index]
                    target_value_prefixs.append(value_prefix)
                else:
                    target_value_prefixs.append(value_prefix)

                value_index += 1

            batch_value_prefixes.append(target_value_prefixs)

        batch_value_prefixes = np.asarray(batch_value_prefixes)
        batch_values = np.zeros_like(batch_value_prefixes)
        batch_variances = np.zeros_like(batch_value_prefixes)
        td_steps_lst = np.ones_like(batch_value_prefixes)
        return batch_value_prefixes, np.asarray(batch_values), td_steps_lst.flatten(), np.asarray(batch_variances), \
               (None, None, None, None)

    def prepare_reward_value(self, traj_lst, transition_pos_lst, indices_lst, collected_transitions, trained_steps, task_idxs=None):
        # value prefix (or reward), value
        batch_value_prefixes, batch_values = [], []
        batch_variances = []
        # search_values = []

        # init
        value_obs_lst, td_steps_lst, value_mask = [], [], []    # mask: 0 -> out of traj
        zero_obs = traj_lst[0].get_zero_obs(self.n_stack, channel_first=False)

        # get obs_{t+k}
        for traj, state_index, idx in zip(traj_lst, transition_pos_lst, indices_lst):
            traj_len = len(traj.reward_lst)     # prev len(traj), TODO: check if td_step log is stable and constant
            # traj_len = len(traj)     # prev len(traj), TODO: check if td_step log is stable and constant
            task_idx = traj.task_idx_lst[state_index] if self.config.data.ind_exp_rp else 0

            # off-policy correction: shorter horizon of td steps
            delta_td = (collected_transitions[task_idx] - idx) // self.auto_td_steps
            if self.value_target in ['mixed', 'max']:
                delta_td = 0
            td_steps = self.td_steps - delta_td
            # td_steps = self.td_steps  # for test off-policy issue
            if not self.episodic:
                td_steps = min(traj_len - state_index, td_steps)
            td_steps = np.clip(td_steps, 1, self.td_steps).astype(np.int32)

            obs_idx = state_index + td_steps

            # prepare the corresponding observations for bootstrapped values o_{t+k}
            traj_obs = traj.get_index_stacked_obs(state_index + td_steps)
            for current_index in range(state_index, state_index + self.unroll_steps + 1):
                if not self.episodic:
                    td_steps = min(traj_len - current_index, td_steps)
                    td_steps = max(td_steps, 1)
                bootstrap_index = current_index + td_steps

                if not self.episodic:
                    if bootstrap_index <= traj_len:
                        value_mask.append(1)
                        beg_index = bootstrap_index - obs_idx
                        end_index = beg_index + self.n_stack
                        obs = traj_obs[beg_index:end_index]
                    else:
                        value_mask.append(0)
                        obs = zero_obs
                else:
                    if bootstrap_index < traj_len:
                        value_mask.append(1)
                        beg_index = bootstrap_index - (state_index + td_steps)
                        end_index = beg_index + self.n_stack
                        obs = traj_obs[beg_index:end_index]
                    else:
                        value_mask.append(0)
                        obs = zero_obs

                value_obs_lst.append(obs)
                td_steps_lst.append(td_steps)

        # reanalyze the bootstrapped value v_{t+k}
        state_lst, value_lst, policy_lst, variance_lst = self.efficient_inference(value_obs_lst, only_value=True, task_idxs=task_idxs)
        batch_size = len(value_lst)
        value_lst = value_lst.reshape(-1) * (np.array([self.discount for _ in range(batch_size)]) ** td_steps_lst)
        value_lst = value_lst * np.array(value_mask)
        # value_lst = np.zeros_like(value_lst)    # for unit test, remove if training
        value_lst = value_lst.tolist()

        # v_{t} = r + ... + gamma ^ k * v_{t+k}
        value_index = 0
        for traj, state_index, idx in zip(traj_lst, transition_pos_lst, indices_lst):
            traj_len = len(traj.reward_lst)  # len(traj)
            # traj_len = len(traj)  # len(traj)
            target_values = []
            target_value_prefixs = []
            target_variances = []

            horizon_id = 0
            value_prefix = 0.0
            for current_index in range(state_index, state_index + self.unroll_steps + 1):
                bootstrap_index = current_index + td_steps_lst[value_index]

                for i, reward in enumerate(traj.reward_lst[current_index:bootstrap_index]):
                    value_lst[value_index] += reward * self.discount ** i

                # reset every lstm_horizon_len
                if horizon_id % self.lstm_horizon_len == 0 and self.value_prefix:
                    value_prefix = 0.0
                horizon_id += 1

                if current_index < traj_len:
                    # Since the horizon is small and the discount is close to 1.
                    # Compute the reward sum to approximate the value prefix for simplification
                    if self.value_prefix:
                        value_prefix += traj.reward_lst[current_index]
                    else:
                        value_prefix = traj.reward_lst[current_index]
                    target_value_prefixs.append(value_prefix)
                else:
                    target_value_prefixs.append(value_prefix)

                if self.episodic:
                    if current_index < traj_len:
                        target_values.append(value_lst[value_index])
                    else:
                        target_values.append(0)
                else:
                    if current_index <= traj_len:
                        target_values.append(value_lst[value_index])
                    else:
                        target_values.append(0)

                target_variances.append(value_lst[value_index])
                value_index += 1

            batch_value_prefixes.append(target_value_prefixs)
            batch_values.append(target_values)
            batch_variances.append(target_variances)

        return np.asarray(batch_value_prefixes), np.asarray(batch_values), np.asarray(td_steps_lst).flatten(), \
               np.asarray(batch_variances), (None, None, None, None)

    def prepare_policy_non_reanalyze(self, traj_lst, transition_pos_lst):
        # policy
        batch_policies = []

        # load searched policy in self-play
        for traj, state_index in zip(traj_lst, transition_pos_lst):
            traj_len = len(traj.reward_lst)
            # traj_len = len(traj)
            target_policies = []

            for current_index in range(state_index, state_index + self.unroll_steps + 1):
                if current_index < traj_len:
                    target_policies.append(traj.policy_lst[current_index])
                else:
                    target_policies.append([0 for _ in range(self.action_space_size)])

            batch_policies.append(target_policies)
        return batch_policies

    def prepare_policy_reanalyze(self, trained_steps, traj_lst, transition_pos_lst, indices_lst, state_lst=None, value_lst=None, policy_lst=None, policy_mask=None, task_idxs=None):
        # policy
        reanalyzed_values = []
        batch_policies = []
        reanalyzed_variances = []

        # init
        if value_lst is None:
            policy_obs_lst, policy_mask = [], []   # mask: 0 -> out of traj
            zero_obs = traj_lst[0].get_zero_obs(self.n_stack, channel_first=False)

            # get obs_{t} instead of obs_{t+k}
            for traj, state_index in zip(traj_lst, transition_pos_lst):
                traj_len = len(traj.reward_lst)
                # traj_len = len(traj)

                game_obs = traj.get_index_stacked_obs(state_index)
                for current_index in range(state_index, state_index + self.unroll_steps + 1):

                    if current_index < traj_len:
                        policy_mask.append(1)
                        beg_index = current_index - state_index
                        end_index = beg_index + self.n_stack
                        obs = game_obs[beg_index:end_index]
                    else:
                        policy_mask.append(0)
                        obs = np.asarray(zero_obs)
                    policy_obs_lst.append(obs)

            # reanalyze the search policy pi_{t}
            state_lst, value_lst, policy_lst, variance_lst = self.efficient_inference(policy_obs_lst, only_value=False, task_idxs=task_idxs)

        # tree search for policies
        batch_size = len(state_lst)

        # temperature
        temperature = self.agent.get_temperature(trained_steps=trained_steps) #* np.ones((batch_size, 1))
        if self.continuous_action:
            r_values, r_policies, best_actions, _, sampled_actions, search_best_indexes, variances_sum = \
                Gumbel_MCTS(self.config).run_multi_continuous(
                    self.model,
                    batch_size, state_lst, value_lst, policy_lst, temperature=temperature, task_idxs=task_idxs
                )
        else:
            r_values, r_policies, best_actions, _, sampled_actions, variances_sum = \
                Gumbel_MCTS(self.config).run_multi_discrete(
                    self.model,
                    batch_size, state_lst, value_lst, policy_lst, temperature=temperature, task_idxs=task_idxs
                )
            search_best_indexes = best_actions

        # concat policy
        policy_index = 0
        policy_masks = []
        mismatch_index = []
        for traj, state_index, ind in zip(traj_lst, transition_pos_lst, indices_lst):
            target_policies = []
            search_values = []
            search_variances = []
            best_action_indices = []
            policy_masks.append([])
            for current_index in range(state_index, state_index + self.unroll_steps + 1):
                traj_len = len(traj.reward_lst)
                # traj_len = len(traj)

                assert (current_index < traj_len) == (policy_mask[policy_index])
                if policy_mask[policy_index]:
                    target_policies.append(r_policies[policy_index])
                    search_values.append(r_values[policy_index])
                    # mask best-action & pi_prime mismatches
                    if r_policies[policy_index].argmax() != search_best_indexes[policy_index]:
                        policy_mask[policy_index] = 0
                        mismatch_index.append(ind + current_index - state_index)
                else:
                    search_values.append(0.0)
                    if self.continuous_action:
                        target_policies.append([0 for _ in range(sampled_actions.shape[1])])
                    else:
                        target_policies.append([0 for _ in range(self.action_space_size)])
                search_variances.append(variances_sum[policy_index])
                policy_masks[-1].append(policy_mask[policy_index])
                best_action_indices.append(search_best_indexes[policy_index])
                policy_index += 1
            batch_policies.append(target_policies)
            reanalyzed_values.append(search_values)
            reanalyzed_variances.append(search_variances)

        if self.rank == 0 and self.config.eval.analysis_value:
            new_log_index = trained_steps // 5000
            if new_log_index > self.last_log_index:
                self.last_log_index = new_log_index
                min_idx = np.asarray(indices_lst).argmin()
                r_value = reanalyzed_values[min_idx][0]
                self.storage.add_log_scalar.remote({
                    'batch_worker/search_value': r_value
                })
        if self.rank == 0:
            new_log_index = trained_steps // 100
            if new_log_index > self.last_log_index:
                self.last_log_index = new_log_index
                self.storage.add_log_distribution.remote({
                    'dist/mismatch_index': np.asarray(mismatch_index)
                })
        policy_masks = np.asarray(policy_masks)
        return (batch_policies, sampled_actions, best_actions, reanalyzed_values, reanalyzed_variances, (state_lst, value_lst, policy_lst, policy_mask), policy_masks)


    def efficient_inference(self, obs_lst, only_value=False, task_idxs=None):
        batch_size = len(obs_lst)
        obs_lst = np.asarray(obs_lst)
        state_lst, value_lst, policy_lst, variance_lst = [], [], [], []
        # split a full batch into slices of mini_infer_size
        mini_batch = self.config.train.mini_batch_size
        slices = np.ceil(batch_size / mini_batch).astype(np.int32)
        with torch.no_grad():
            for i in range(slices):
                beg_index = mini_batch * i
                end_index = mini_batch * (i + 1)
                current_obs = obs_lst[beg_index:end_index]
                current_task_idxs = torch.from_numpy(task_idxs[beg_index:end_index]).long().cuda()
                current_obs = formalize_obs_lst(current_obs, self.image_based)
                # obtain the statistics at current steps
                with autocast():
                    states, values, policies, variances = self.model.initial_inference(current_obs, task_idxs=current_task_idxs)

                values = values.detach().cpu().numpy().flatten()
                variances = variances.detach().cpu().numpy().flatten()

                # concat
                value_lst.append(values)
                variance_lst.append(variances)
                if not only_value:
                    state_lst.append(states)
                    policy_lst.append(policies)

        value_lst = np.concatenate(value_lst)
        variance_lst = np.concatenate(variance_lst)
        if not only_value:
            state_lst = torch.cat(state_lst)
            policy_lst = torch.cat(policy_lst)
        return state_lst, value_lst, policy_lst, variance_lst


# ======================================================================================================================
# batch worker
# ======================================================================================================================
def start_batch_worker(rank, agent, replay_buffer, storage, batch_storage, config, seed=None):
    """
    Start a GPU batch worker. Call this method remotely.
    """
    worker = BatchWorker.remote(rank, agent, replay_buffer, storage, batch_storage, config, seed=seed)
    print(f"[Batch worker GPU] Starting batch worker GPU {rank} at process {os.getpid()}.")
    worker.run.remote()

def start_batch_worker_cpu(rank, agent, replay_buffer, storage, prebatch_storage, config):
    worker = BatchWorker_CPU.remote(rank, agent, replay_buffer, storage, prebatch_storage, config)
    print(f"[Batch worker CPU] Starting batch worker CPU {rank} at process {os.getpid()}.")
    worker.run.remote()

def start_batch_worker_gpu(rank, agent, replay_buffer, storage, prebatch_storage, batch_storage, config):
    worker = BatchWorker_GPU.remote(rank, agent, replay_buffer, storage, prebatch_storage, batch_storage, config)
    print(f"[Batch worker GPU] Starting batch worker GPU {rank} at process {os.getpid()}.")
    worker.run.remote()