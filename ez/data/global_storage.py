import copy
import os
import time
import ray
import numpy as np
from collections import deque

@ray.remote
class GlobalStorage:
    def __init__(self, self_play_model, reanalyze_model, latest_model):
        self.models = {
            'self_play': self_play_model,
            'reanalyze': reanalyze_model,
            'latest': latest_model
        }
        self.log_scalar = {}
        self.eval_log_scalar = {}
        self.log_distribution = {}
        self.counter = 0
        self.eval_counter = 0
        self.best_score = - np.inf
        self.start = False
        self.variance_max = None # -1e6
        self.variance_min = None # 1e6
        # self.batch = None
        self.epi_return_history = {}
        self.saturated_flag = {}

    def get_weights(self, model_name):
        assert model_name in self.models.keys()
        return self.models[model_name].get_weights()

    def set_weights(self, weights, model_name):
        assert model_name in self.models.keys()
        # print('[Update] set recent model of {}'.format(model_name))
        return self.models[model_name].set_weights(weights, hard=True)

    def increase_counter(self):
        self.counter += 1

    def get_counter(self):
        return self.counter

    def set_eval_counter(self, counter):
        self.eval_counter = counter

    def get_eval_counter(self):
        return self.eval_counter

    def set_start_signal(self):
        self.start = True

    def get_start_signal(self):
        return self.start

    def update_variance_minmax(self, variance, task_idxs):
        if self.variance_max is None:
            task_num = int(task_idxs.max() + 1)
            self.variance_max = -1e6 * np.ones(task_num)
            self.variance_min = 1e6 * np.ones(task_num)
        for i, task_idx in enumerate(task_idxs):
            self.variance_max[task_idx] = max(variance[i], self.variance_max[task_idx])
            self.variance_min[task_idx] = min(variance[i], self.variance_min[task_idx])

    def get_variance_minmax(self):
        return self.variance_max, self.variance_min

    def init_per_task_max_return(self, values):
        self.per_task_max_returns = values

    def update_per_task_max_return(self, values, task_idxs):
        for task_idx in task_idxs:
            self.per_task_max_returns[task_idx] = max(values[task_idx], self.per_task_max_returns[task_idx])

    def get_per_task_max_return(self):
        return self.per_task_max_returns

    # def set_batch(self, batch):
    #     self.batch = batch
    #
    # def get_batch(self):
    #     batch = copy.deepcopy(self.batch)
    #     self.batch = None
    #     return batch

    def set_best_score(self, score):
        self.best_score = max(self.best_score, score)

    def get_best_score(self):
        return self.best_score

    def add_log_scalar(self, dic):
        for key, val in dic.items():
            if key not in self.log_scalar.keys():
                self.log_scalar[key] = []

            self.log_scalar[key].append(val)

    def add_epi_return_history(self, dic):
        key, value = next(iter(dic.items()))
        if key not in self.saturated_flag.keys():
            self.saturated_flag[key] = False
        if key not in self.epi_return_history.keys():
            self.epi_return_history[key] = deque(maxlen=10)
        self.epi_return_history[key].append(value)

    def get_saturated(self, task_name):
        if len(self.epi_return_history[task_name]) > 9 and not self.saturated_flag[task_name]:
            if (max(self.epi_return_history[task_name]) - min(self.epi_return_history[task_name]) < max(
                self.epi_return_history[task_name]) * 0.1) and (min(self.epi_return_history[task_name]) > 60):
                self.saturated_flag[task_name] = True
        return self.saturated_flag[task_name]

    def add_eval_log_scalar(self, dic):
        for key, val in dic.items():
            if key not in self.eval_log_scalar.keys():
                self.eval_log_scalar[key] = []
            self.eval_log_scalar[key].append(val)

    def add_log_distribution(self, dic):
        for key, val in dic.items():
            if key not in self.log_distribution.keys():
                self.log_distribution[key] = []

            self.log_distribution[key] += val.tolist()

    def get_log(self):
        # for scalar
        scalar = {}
        for key, val in self.log_scalar.items():
            scalar[key] = np.mean(val)

        eval_scalar = {}
        for key, val in self.eval_log_scalar.items():
            eval_scalar[key] = np.mean(val)

        # for distribution
        distribution = {}
        for key, val in self.log_distribution.items():
            distribution[key] = np.array(val).flatten()

        self.log_scalar = {}
        self.eval_log_scalar = {}
        self.log_distribution = {}
        return eval_scalar, scalar, distribution

# ======================================================================================================================
# global storage server
# ======================================================================================================================
