import copy
import os
import time
# import SMOS
import ray
import torch
import numpy as np
import random

from torch.nn import L1Loss
from pathlib import Path
from torch.cuda.amp import autocast as autocast

from ez.worker.base import Worker
from ez.envs import make_envs
from ez.utils.format import formalize_obs_lst, DiscreteSupport, allocate_gpu, prepare_obs_lst, symexp, set_seed
from ez.data.replay_buffer import get_replay_buffer
from ez.mcts.cy_mcts import Gumbel_MCTS
from collections import deque

@ray.remote(num_gpus=0.05)
class DataWorker(Worker):
    def __init__(self, rank, agent, replay_buffer, storage, config, seed=None):
        super().__init__(rank, agent, replay_buffer, storage, config)

        self.model_update_interval = config.train.self_play_update_interval
        self.traj_pool = []
        self.pool_size = 1

        if seed is not None:
            self.seed = seed
            set_seed(seed)

        self.multi_task = config.env.multi_task
        self.env = config.env.env
        self.continuous_action = config.env.continuous_action

        # time.sleep(10000)     # for ray ipdb debug

    @torch.no_grad()
    def run(self):
        config = self.config

        # create the model for self-play data collection
        self.model = self.agent.build_model()
        self.model.cuda()
        if int(torch.__version__[0]) == 2:
            self.model = torch.compile(self.model)
        self.model.eval()
        self.resume_model()

        # make env
        num_envs = config.data.num_envs
        save_path = Path(config.save_path)
        if config.data.save_video:
            video_path = save_path / 'self_play_videos'
        else:
            video_path = None
        cur_seed = self.seed

        if self.multi_task:
            envs, self.action_masks = [], []
            for _ in range(num_envs):
                envs += make_envs(self.env, config.env.game, num_envs, cur_seed + self.rank * num_envs,
                                  save_path=video_path, episodic_life=config.env.episodic, **config.env)  # prev episodic_life=True
                self.action_masks += config.env.action_masks
            task_num = self.config.env.task_num
            task_idxs = np.asarray([i % task_num for i in range(len(envs))]).astype(int)
            print(f'task_idxs: {task_idxs}')
            self.storage.init_per_task_max_return.remote([0.0 for _ in range(len(envs))])
        else:
            envs = make_envs(self.env, config.env.game, num_envs, cur_seed + self.rank * num_envs, save_path=video_path,
                             episodic_life=config.env.episodic, **config.env)
            task_idxs = np.asarray([0 for _ in range(len(envs))])
        num_envs = len(envs)

        # initialization
        trained_steps = 0           # current training steps
        collected_transitions = [0 for _ in range(self.config.env.task_num)]
        epi_return_history = [deque(maxlen=10) for _ in range(self.config.env.task_num)]
        start_training = False      # is training
        max_transitions = config.data.total_transitions // config.actors.data_worker  # max transitions to collect in this worker
        print(f'total:{config.data.total_transitions}, workers={config.actors.data_worker}, max_transitions: {max_transitions}')
        dones = [False for _ in range(num_envs)]
        traj_len = [0 for _ in range(num_envs)]
        episode_lens = [0 for _ in range(num_envs)]

        stack_obs_windows, game_trajs = self.agent.init_envs(envs, max_steps=self.config.data.trajectory_size, seed=cur_seed)
        prev_game_trajs = [None for _ in range(num_envs)]  # previous game trajectories (split a full game trajectory into several sub trajectories)

        # log data
        episode_return = [0. for _ in range(num_envs)]

        # while loop for collecting data
        prev_train_steps = 0
        cnt = 0
        gumbel_mcts = Gumbel_MCTS(config)
        while not self.is_finished(trained_steps):
            trained_steps = ray.get(self.storage.get_counter.remote())
            if not start_training:
                start_training = ray.get(self.storage.get_start_signal.remote())

            # get the fresh model weights
            self.get_recent_model(trained_steps, 'self_play')

            if self.env != 'Atari':
                if collected_transitions[0] > max_transitions:
                    time.sleep(10)
                    continue
                # self-play is faster than training speed or finished
                if start_training and (collected_transitions[0] / max_transitions) > (trained_steps / self.config.train.training_steps):
                    time.sleep(1)
                    continue

            if self.config.ray.single_process:
                trained_steps = ray.get(self.storage.get_counter.remote())
                if start_training and trained_steps <= prev_train_steps:
                    time.sleep(0.1)
                    continue
                prev_train_steps = trained_steps

            # print('self-playing')
            # temperature
            temperature = self.agent.get_temperature(trained_steps=trained_steps)

            # stack obs
            current_stacked_obs = formalize_obs_lst(stack_obs_windows, image_based=config.env.image_based)
            # obtain the statistics at current steps
            with autocast():
                states, values, policies, _ = self.model.initial_inference(current_stacked_obs,
                                                                           task_idxs=torch.from_numpy(task_idxs).long().cuda())

            # process outputs
            values = values.detach().cpu().numpy().flatten()

            # tree search for policies
            if self.continuous_action:
                r_values, r_policies, best_actions, _, sampled_actions, best_indexes, variances = \
                    gumbel_mcts.run_multi_continuous(
                        self.model, num_envs, states, values, policies, temperature=temperature,
                        task_idxs=task_idxs,
                    )
            else:
                r_values, r_policies, best_actions, _, sampled_actions, variances, _ = \
                    gumbel_mcts.run_multi_discrete(
                        self.model, num_envs, states, values, policies, temperature=temperature, task_idxs=task_idxs
                    )

            if cnt % 20 == 0:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            cnt += 1

            # step action in environments
            for i in range(num_envs):
                task_idx = int(task_idxs[i])
                task_name = self.config.env.env_names[int(task_idx)] if self.multi_task else self.config.env.game

                if self.env == 'Atari':
                    if collected_transitions[task_idx] > max_transitions:
                        # time.sleep(1)
                        continue
                    # self-play is faster than training speed or finished
                    if start_training and (collected_transitions[task_idx] / max_transitions) > (trained_steps / self.config.train.training_steps):
                        # time.sleep(1)
                        continue

                action = best_actions[i]
                action_to_apply = action
                if self.multi_task:
                    if self.env == 'Atari':
                        assert action_to_apply < int(sum(self.action_masks[i])), f'action out of range'
                    if self.continuous_action:
                        action_to_apply = action[:int(sum(self.action_masks[i]))]
                if not start_training:
                    action_to_apply = envs[i].action_space.sample()
                obs, reward, done, info = envs[i].step(action_to_apply)
                if self.multi_task:
                    if not self.config.env.image_based:
                        obs = np.concatenate([obs, np.zeros(self.config.env.obs_shape - obs.shape[0])]).astype(np.float32)     # pad to same dim with 0
                dones[i] = done
                traj_len[i] += 1
                episode_lens[i] += 1
                episode_return[i] += info['raw_reward']

                # save data to trajectory buffer
                game_trajs[i].store_search_results(values[i], r_values[i], r_policies[i])
                game_trajs[i].append(action, obs, reward, task_idx=task_idxs[i])
                if self.env == 'Atari':
                    game_trajs[i].snapshot_lst.append([])
                else:
                    game_trajs[i].snapshot_lst.append([])

                # fresh stack windows
                del stack_obs_windows[i][0]
                stack_obs_windows[i].append(obs)

                # if current trajectory is full; we will save the previous trajectory
                if game_trajs[i].is_full() and not dones[i]:
                    if prev_game_trajs[i] is not None:
                        collected_transitions[task_idx] += len(prev_game_trajs[i])
                        self.save_previous_trajectory(i, prev_game_trajs, game_trajs, task_idxs[i], padding=True)

                    prev_game_trajs[i] = game_trajs[i]

                    # new trajectory
                    game_trajs[i] = self.agent.new_game(max_steps=self.config.data.trajectory_size)
                    game_trajs[i].init(stack_obs_windows[i])

                    traj_len[i] = 0

                # reset an env if done
                if dones[i]:
                    # save the previous trajectory
                    if prev_game_trajs[i] is not None:
                        collected_transitions[task_idx] += len(prev_game_trajs[i])
                        self.save_previous_trajectory(i, prev_game_trajs, game_trajs, task_idxs[i], padding=True)

                    if len(game_trajs[i]) > 0 and self.config.env.episodic:
                        # save current trajectory
                        collected_transitions[task_idx] += len(game_trajs[i])
                        game_trajs[i].pad_over([], [], [], [], [])
                        game_trajs[i].save_to_memory()
                        self.put_trajs(game_trajs[i], task_idxs[i])

                    # log
                    if self.multi_task:
                        self.storage.add_log_scalar.remote({
                            f'self_play/{task_name}_episode_return': episode_return[i],
                        })
                    else:
                        self.storage.add_log_scalar.remote({
                            'self_play/episode_return': episode_return[i],
                        })
                    self.storage.add_log_scalar.remote({
                        f'self_play/{task_name}_episode_len': episode_lens[i],
                        'self_play/temperature': temperature,
                    })
                    self.storage.add_epi_return_history.remote({
                        task_name: episode_return[i]
                    })

                    # reset the finished env and new a env
                    stacked_obs, traj = self.agent.init_env(envs[i], max_steps=self.config.data.trajectory_size, seed=cur_seed)
                    if self.continuous_action:
                        saturated = ray.get(self.storage.get_saturated.remote(task_name))
                        if (trained_steps > int(self.config.train.training_steps * 0.4)) or saturated:
                            skip = self.config.env.n_skip
                            envs[i]._max_episode_steps = min(self.config.env.max_episode_steps * 2, 1000) // skip
                            print(f'finish changing episode len, now {envs[i]._max_episode_steps}')

                    stack_obs_windows[i] = stacked_obs
                    game_trajs[i] = traj
                    prev_game_trajs[i] = None

                    epi_return_history[task_idx].append(episode_return[i])


                    traj_len[i] = 0
                    episode_lens[i] = 0
                    episode_return[i] = 0
                    gumbel_mcts.gumbel_noise = None


    def save_previous_trajectory(self, idx, prev_game_trajs, game_trajs, task_idx, padding=True):
        """put the previous game trajectory into the pool if the current trajectory is full
        Parameters
        ----------
        idx: int
            index of the traj to handle
        prev_game_trajs: list
            list of the previous game trajectories
        game_trajs: list
            list of the current game trajectories
        """
        if padding:
            # pad over last block trajectory
            if self.config.model.value_target == 'bootstrapped':
                gap_step = self.config.env.n_stack + self.config.rl.td_steps + self.config.rl.unroll_steps + 1
            elif self.config.model.value_target == 'GAE':
                # extra = max(0, min(int(1 / (1 - self.config.rl.td_lambda)), self.config.model.GAE_max_steps) - self.config.rl.unroll_steps)
                # extra = max(0, int(1 / (1 - self.config.rl.td_lambda)) - self.config.rl.unroll_steps - 1)
                extra = min(int(1 / (1 - self.config.rl.td_lambda)), self.config.model.GAE_max_steps)
                gap_step = self.config.env.n_stack + 1 + extra + 1

            beg_index = self.config.env.n_stack
            if self.config.model.value_target == 'bootstrapped':
                end_index = beg_index + self.config.rl.unroll_steps + 1 + self.config.rl.td_steps
            elif self.config.model.value_target == 'GAE':
                end_index = beg_index + gap_step
            else:
                raise NotImplementedError
            pad_obs_lst = game_trajs[idx].obs_lst[beg_index:end_index]

            pad_policy_lst = game_trajs[idx].policy_lst[0:self.config.rl.unroll_steps + 1 + self.config.rl.td_steps]
            pad_reward_lst = game_trajs[idx].reward_lst[0:gap_step-1]
            pad_pred_values_lst = game_trajs[idx].pred_value_lst[0:gap_step]
            pad_search_values_lst = game_trajs[idx].search_value_lst[0:gap_step]
            pad_prior_lst = game_trajs[idx].policy_prior_lst[0:gap_step]
            pad_task_idx_lst = game_trajs[idx].task_idx_lst[0:gap_step]

            # pad over and save
            if self.continuous_action:
                prev_game_trajs[idx].pad_over(pad_obs_lst, pad_reward_lst, pad_pred_values_lst, pad_search_values_lst,
                                              pad_policy_lst, tail_priors=pad_prior_lst,
                                              tail_task_idxs=pad_task_idx_lst)
            else:
                prev_game_trajs[idx].pad_over(pad_obs_lst, pad_reward_lst, pad_pred_values_lst, pad_search_values_lst,
                                              pad_policy_lst, tail_task_idxs=pad_task_idx_lst)


        prev_game_trajs[idx].save_to_memory()
        self.put_trajs(prev_game_trajs[idx], task_idx)

        # reset last block
        prev_game_trajs[idx] = None

    def put_trajs(self, traj, task_idx):
        # if self.config.priority.use_priority:
        #     traj_len = len(traj)
        #     pred_values = torch.from_numpy(np.array(traj.pred_value_lst)).cuda().float()
        #     # search_values = torch.from_numpy(np.array(traj.search_value_lst)).cuda().float()
        #     if self.config.model.value_target == 'bootstrapped':
        #         target_values = torch.from_numpy(np.asarray(traj.get_bootstrapped_value())).cuda().float()
        #     elif self.config.model.value_target == 'GAE':
        #         target_values = torch.from_numpy(np.asarray(traj.get_gae_value())).cuda().float()
        #     else:
        #         raise NotImplementedError
        #     # priorities = L1Loss(reduction='none')(pred_values, search_values).detach().cpu().numpy() + self.config.priority.min_prior
        #     priorities = L1Loss(reduction='none')(pred_values[:traj_len], target_values[:traj_len]).detach().cpu().numpy() + self.config.priority.min_prior
        #     # priorities = priorities.clip(0, 1)
        #     # print(f'max_prior={priorities.max()}')
        # else:
        priorities = None
        self.traj_pool.append(traj)
        # save the game histories and clear the pool
        if len(self.traj_pool) >= self.pool_size:
            self.replay_buffer.save_pools.remote(self.traj_pool, priorities, task_idx)
            del self.traj_pool[:]

# ======================================================================================================================
# data worker for self-play
# ======================================================================================================================
def start_data_worker(rank, agent, replay_buffer, storage, config, seed=None):
    """
    Start a data worker. Call this method remotely.
    """
    data_worker = DataWorker.remote(rank, agent, replay_buffer, storage, config, seed=seed)
    data_worker.run.remote()
    print(f'[Data worker] Start data worker {rank} at process {os.getpid()}.')
