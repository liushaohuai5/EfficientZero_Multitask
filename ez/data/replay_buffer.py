import os
import time
import numpy as np
import ray
import pickle

from ez.utils.format import set_seed

@ray.remote
class ReplayBuffer:
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size')
        self.buffer_size = kwargs.get('buffer_size')
        self.top_transitions = kwargs.get('top_transitions')
        self.use_priority = kwargs.get('use_priority')
        self.continuous_action = kwargs.get('continuous_action')
        self.env = kwargs.get('env')
        self.total_transitions = kwargs.get('total_transitions')
        self.multi_task = kwargs.get('multi_task')
        self.task_num = kwargs.get('task_num')
        self.ind_exp_rp = kwargs.get('ind_exp_rp')
        # self.seed = kwargs.get('seed')
        # set_seed(self.seed)

        self.base_idx = 0
        self.clear_time = 0
        if not self.ind_exp_rp:
            self.task_num = 1
        self.buffer = [[] for _ in range(self.task_num)]
        self.priorities = [[] for _ in range(self.task_num)]
        # self.snapshots = []
        self.transition_idx_look_up = [[] for _ in range(self.task_num)]

    def save_pools(self, traj_pool, priorities, task_idx):
        # save a list of game histories
        for traj in traj_pool:
            if len(traj) > 0:
                self.save_trajectory(traj, priorities, task_idx)

    def save_trajectory(self, traj, priorities, task_idx):
        traj_len = len(traj)
        if not self.ind_exp_rp:
            task_idx = 0
        if priorities is None:
            max_prio = self.priorities[task_idx].max() if len(self.buffer[task_idx])>0 else 1
            # print(f'new_max_prio={max_prio}')
            self.priorities[task_idx] = np.concatenate((self.priorities[task_idx], [max_prio for _ in range(traj_len)]))
        else:
            assert len(traj) == len(priorities), " priorities should be of same length as the game steps"
            priorities = priorities.copy().reshape(-1)
            # self.priorities = np.concatenate((self.priorities, priorities))
            max_prio = self.priorities[task_idx].max() if len(self.buffer[task_idx])>0 else 1
            self.priorities[task_idx] = np.concatenate((self.priorities[task_idx], [max(max_prio, priorities[i]) for i in range(traj_len)]))
            # self.priorities[idx] = np.concatenate((self.priorities[idx], [max(max_prio, priorities.max()) for i in range(traj_len)]))

        # for snapshot in traj.snapshot_lst:
        #     self.snapshots.append(snapshot)

        self.buffer[task_idx].append(traj)
        self.transition_idx_look_up[task_idx] += [(self.base_idx + len(self.buffer[task_idx]) - 1, step_pos) for step_pos in range(traj_len)]


    def get_item(self, idx, task_idx):
        if not self.ind_exp_rp:
            task_idx = 0
        traj_idx, state_index = self.transition_idx_look_up[task_idx][idx]
        traj_idx -= self.base_idx
        traj = self.buffer[task_idx][traj_idx]

        return traj, state_index

    def prepare_batch_context(self, batch_size, alpha, beta, rank, cnt):
        # if rank == 0 and cnt % 100 == 0:
        #     from line_profiler import LineProfiler
        #     lp = LineProfiler()
        #     lp_wrapper = lp(self._prepare_batch_context)
        #     batch_context = lp_wrapper(batch_size, alpha, beta)
        #     lp.print_stats()
        # else:

        batch_context = self._prepare_batch_context(batch_size, alpha, beta)
        batch_context = (batch_context, False)

        # for supervised learning
        # if cnt % 200 == 0:
        #     batch_context = self._prepare_batch_context_supervised(batch_size, is_validation=True)
        #     batch_context = (batch_context, True)
        # else:
        #     batch_context = self._prepare_batch_context_supervised(batch_size, alpha=alpha, beta=beta,
        #                                                            is_validation=False, force_uniform=True)
        #     batch_context = (batch_context, False)

        return batch_context

    # def _prepare_batch_context_supervised(self, batch_size, alpha=None, beta=None, is_validation=False, force_uniform=False):
    #     transition_num = self.get_transition_num()
    #     if is_validation:
    #         validation_set = np.arange(int(transition_num * 0.95), transition_num)
    #         indices_lst = np.random.choice(validation_set, batch_size, replace=False)
    #         weights_lst = (1 / batch_size) * np.ones_like(indices_lst)
    #     else:
    #         # sample data
    #         if self.use_priority:
    #             probs = self.priorities ** alpha
    #         else:
    #             probs = np.ones_like(self.priorities)
    #         probs = probs[:int(0.95 * transition_num)]
    #         probs = probs / probs.sum()
    #
    #         training_set = np.arange(int(transition_num * 0.95))
    #         if force_uniform:
    #             indices_lst = np.random.choice(training_set, batch_size, replace=False)
    #             weights_lst = (1 / batch_size) * np.ones_like(indices_lst)
    #         else:
    #             indices_lst = np.random.choice(training_set, batch_size, p=probs, replace=False)
    #             weights_lst = (transition_num * probs[indices_lst]) ** (-beta)
    #             weights_lst = weights_lst / weights_lst.max()
    #
    #     traj_lst, transition_pos_lst = [], []
    #     # obtain the
    #     for idx in indices_lst:
    #         traj, state_index = self.get_item(idx)
    #         traj_lst.append(traj)
    #         transition_pos_lst.append(state_index)
    #
    #     make_time_lst = [time.time() for _ in range(len(indices_lst))]
    #     context = [self.split_trajs(traj_lst), transition_pos_lst, indices_lst, weights_lst, make_time_lst,
    #                transition_num, self.priorities[indices_lst]]
    #     return context


    def _prepare_batch_context(self, batch_size, alpha, beta):

        mini_batch_size = batch_size // self.task_num
        traj_lst, transition_pos_lst, indices_lst, weights_lst, make_time_lst, prior_lst = [], [], [], [], [], []
        if not self.multi_task:
            assert self.task_num == 1, 'single task training and task num should be 1 rather than {}'.format(self.task_num)
        for task_idx in range(self.task_num):
            transition_num = self.get_transition_num(task_idx)

            # sample data
            if self.use_priority:
                probs = self.priorities[task_idx] ** alpha
            else:
                probs = np.ones_like(self.priorities[task_idx])

            top_transitions = self.top_transitions

            # sample the top transitions of the current buffer
            if self.continuous_action and len(self.priorities[task_idx]) > top_transitions:
                idx = int(len(self.priorities[task_idx]) - top_transitions)
                probs[:idx] = 0
                # self.priorities[:idx] = 0

            probs = probs / probs.sum()

            indices = np.random.choice(transition_num, mini_batch_size, p=probs, replace=False)

            # weight
            weights = (transition_num * probs[indices]) ** (-beta)
            weights = weights / (weights.max() + 1e-6)
            weights = weights.clip(0.1, 1)    # TODO: try weights clip, prev 0.1

            # obtain the
            for idx in indices:
                traj, state_index = self.get_item(idx, task_idx)
                traj_lst.append(traj)
                transition_pos_lst.append(state_index)

            indices_lst += indices.tolist()
            weights_lst += weights.tolist()
            make_time_lst += [time.time() for _ in range(len(indices_lst))]
            prior_lst += self.priorities[task_idx][indices].tolist()

        context = [self.split_trajs(traj_lst), transition_pos_lst, indices_lst, weights_lst, make_time_lst, transition_num, prior_lst]
        return context

    def split_trajs(self, traj_lst):
        obs_lsts, reward_lsts, policy_lsts, policy_prior_lsts, action_lsts, pred_value_lsts, search_value_lsts, \
        bootstrapped_value_lsts, snapshot_lsts, task_idx_lsts = [], [], [], [], [], [], [], [], [], []
        for traj in traj_lst:
            obs_lsts.append(traj.obs_lst)
            reward_lsts.append(traj.reward_lst)
            policy_lsts.append(traj.policy_lst)
            policy_prior_lsts.append(traj.policy_prior_lst)
            action_lsts.append(traj.action_lst)
            pred_value_lsts.append(traj.pred_value_lst)
            search_value_lsts.append(traj.search_value_lst)
            bootstrapped_value_lsts.append(traj.bootstrapped_value_lst)
            snapshot_lsts.append(traj.snapshot_lst)
            task_idx_lsts.append(traj.task_idx_lst)
        return [obs_lsts, reward_lsts, policy_lsts, policy_prior_lsts, action_lsts, pred_value_lsts, search_value_lsts, bootstrapped_value_lsts,
                # snapshot_lsts,
                task_idx_lsts
                ]

    def update_root_values(self, batch_indices, search_values, transition_positions, unroll_steps):
        val_idx = 0
        for idx, pos in zip(batch_indices, transition_positions):
            traj_idx, state_index = self.transition_idx_look_up[idx]
            traj_idx -= self.base_idx
            for i in range(unroll_steps + 1):
                self.buffer[traj_idx].search_value_lst.setflags(write=True)
                if pos + i < len(self.buffer[traj_idx].search_value_lst):
                    self.buffer[traj_idx].search_value_lst[pos + i] = search_values[val_idx][i]
                    # self.priorities[idx] = np.abs(
                    #     self.buffer[traj_idx].pred_value_lst[pos + i] - self.buffer[traj_idx].search_value_lst[pos + i]
                    # )
            val_idx += 1

    def update_priorities(self, batch_indices, batch_priorities, make_time, task_indices, mask=None):
        # update the priorities for data still in replay buffer
        if mask is None:
            mask = np.ones(len(batch_indices))
        if not self.ind_exp_rp:
            task_indices = np.zeros_like(task_indices).astype(np.int)
        for i in range(len(batch_indices)):
            # if make_time[i] > self.clear_time:
            assert make_time[i] > self.clear_time, f'make_time {make_time[i]} <= clear_time {self.clear_time}'
            idx, prio, task_idx = batch_indices[i], batch_priorities[i], task_indices[i]
            if mask[i] == 1:
                self.priorities[task_idx][idx] = prio

    def get_priorities(self):
        priorities = []
        for task_idx in range(self.task_num):
            priorities.append(self.priorities[task_idx].tolist())
        return priorities

    # def get_snapshots(self, indices_lst):
    #     selected_snapshots = []
    #     for idx in indices_lst:
    #         selected_snapshots.append(self.snapshots[idx])
    #     return selected_snapshots

    def get_traj_num(self):
        traj_num = []
        for task_idx in range(self.task_num):
            traj_num.append(len(self.buffer[task_idx]))
        return traj_num

    def get_transition_num(self, task_idx=None):
        if task_idx is None:
            transition_num = []
            for task_idx in range(self.task_num):
                assert len(self.transition_idx_look_up[task_idx]) == len(self.priorities[task_idx])
                # assert len(self.priorities) == len(self.snapshots)
                transition_num.append(len(self.transition_idx_look_up[task_idx]))
        else:
            assert len(self.transition_idx_look_up[task_idx]) == len(self.priorities[task_idx])
            transition_num = len(self.transition_idx_look_up[task_idx])
        return transition_num

    def save_buffer(self):
        path = '/workspace/EZ-Codebase/buffer/'
        f_buffer = open(path + 'buffer.b', 'wb')
        pickle.dump(self.buffer, f_buffer)
        f_buffer.close()
        f_priorities = open(path + 'priorities.b', 'wb')
        pickle.dump(self.priorities, f_priorities)
        f_priorities.close()
        f_lookup = open(path + 'lookup.b', 'wb')
        pickle.dump(self.transition_idx_look_up, f_lookup)
        f_lookup.close()
        # f_snapshot = open(path + 'snapshots.b', 'wb')
        # pickle.dump(self.snapshots, f_snapshot)
        # f_snapshot.close()
        return True

    def load_buffer(self):
        path = '/workspace/EZ-Codebase/buffer/'
        f = open(path + 'buffer.b', 'rb')
        self.buffer = pickle.load(f)
        f.close()
        f = open(path + 'priorities.b', 'rb')
        self.priorities = pickle.load(f)
        f.close()
        f = open(path + 'lookup.b', 'rb')
        self.transition_idx_look_up = pickle.load(f)
        f.close()
        # f = open(path + 'snapshots.b', 'rb')
        # self.snapshots = pickle.load(f)
        # f.close()
        return True

# ======================================================================================================================
# replay buffer server
# ======================================================================================================================
def start_replay_buffer_server(manager, replay_buffer):
    """
    Start a replay buffer in current process. Call this method remotely.
    """
    # initialize replay buffer
    start_server(manager.replay_buffer_connection, manager.replay_buffer_register_name, replay_buffer)


def get_replay_buffer(manager):
    """
    Get connection to a replay buffer server.
    """
    return get_remote_object(manager.replay_buffer_connection, manager.replay_buffer_register_name)
