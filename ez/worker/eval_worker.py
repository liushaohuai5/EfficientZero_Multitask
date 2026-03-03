import os
import time
# import SMOS
import ray
import torch
import logging
import numpy as np
import gym

from pathlib import Path
from torch.cuda.amp import autocast as autocast

from .base import Worker
from ez.envs import make_envs
from ez.eval import eval_pure

@ray.remote(num_gpus=0.05, num_cpus=0.5)
# @ray.remote(num_gpus=0.2)
class EvalWorker(Worker):
    def __init__(self, agent, replay_buffer, storage, config):
        super().__init__(0, agent, replay_buffer, storage, config)
        self.env = config.env
        self.multi_task = config.env.multi_task
        if self.multi_task:
            self.action_masks = config.env.action_masks
        else:
            self.action_masks = None

        # time.sleep(10000)
        # set_seed(config.env.base_seed)

    def run(self):
        model = self.agent.build_model()
        if int(torch.__version__[0]) == 2:
            model = torch.compile(model)
        best_eval_score = float('-inf')
        episodes = 0
        counter = 0
        eval_steps = 1000 if self.config.env.continuous_action else 27000           # due to time limitation, eval 3000 steps (instead of 27000) during training.
        save_path = Path(self.config.save_path) / 'evaluation' / 'step_{}'.format(counter)
        save_path.mkdir(parents=True, exist_ok=True)
        if save_path is not None:
            video_path = save_path / 'recordings'
            video_path.mkdir(parents=True, exist_ok=True)
        else:
            video_path = None
        self.config.env.max_episode_steps = eval_steps

        if self.multi_task:
            envs, action_masks = [], []
            num_runs = 1
            if self.config.env.env == 'ManiSkill':
                num_runs = 10
            for _ in range(num_runs):
                envs += make_envs(self.env.env, self.config.env.game, self.config.train.eval_n_episode, self.config.env.base_seed,
                                  save_path=video_path, episodic_life=False, render_mode='rgb_array', **self.env)  # prev episodic_life=True
                action_masks += self.config.env.action_masks
            task_num = self.config.env.task_num
            task_idxs = np.asarray([i % task_num for i in range(len(envs))]).astype(int)
        else:
            envs = make_envs(self.env.env, self.config.env.game, self.config.train.eval_n_episode, self.config.env.base_seed, save_path=video_path,
                             episodic_life=False, render_mode='rgb_array', **self.env)
            task_idxs = np.asarray([0 for _ in range(len(envs))])

        while not self.is_finished(counter):
            counter = ray.get(self.storage.get_counter.remote())
            if counter >= self.config.train.eval_interval * episodes:
                print('[Eval] Start evaluation at step {}.'.format(counter))

                episodes += 1
                model.set_weights(ray.get(self.storage.get_weights.remote('self_play')), hard=True)
                model.eval()

                save_path = Path(self.config.save_path) / 'evaluation' / 'step_{}'.format(counter)
                save_path.mkdir(parents=True, exist_ok=True)
                model_path = Path(self.config.save_path) / 'model.p'
                eval_score, eval_len, infos = eval_pure(self.agent, model, self.config.train.eval_n_episode, save_path, self.config, envs,
                                                 max_steps=eval_steps, use_pb=False, verbose=0, task_idxs=task_idxs)

                if self.config.env.multi_task:
                    env_names = self.config.env.env_names
                    per_task_score, per_task_len = np.zeros(self.config.env.task_num), np.zeros(self.config.env.task_num)
                    for i, (score, length) in enumerate(zip(eval_score, eval_len)):
                        task_idx = task_idxs[i]
                        per_task_score[task_idx] += score
                        per_task_len[task_idx] += length

                    log_scalers = {}
                    for task_idx, name in enumerate(env_names):
                        log_scalers[f'eval/{name}'] = per_task_score[task_idx] / num_runs
                        if self.config.env.env == 'ManiSkill':
                            log_scalers[f'eval/{name}_success_once'] = infos[name]['success_once'] / num_runs
                            log_scalers[f'eval/{name}_success_at_end'] = infos[name]['success_at_end'] / num_runs
                    self.storage.set_eval_counter.remote(counter)
                    self.storage.add_eval_log_scalar.remote(log_scalers)
                else:
                    mean_score = eval_score.mean()
                    std_score = eval_score.std()
                    min_score = eval_score.min()
                    max_score = eval_score.max()

                    if mean_score >= best_eval_score:
                        best_eval_score = mean_score
                        self.storage.set_best_score.remote(best_eval_score)
                        torch.save(model.state_dict(), model_path)

                    self.storage.set_eval_counter.remote(counter)
                    self.storage.add_eval_log_scalar.remote({
                        'eval/mean_score': mean_score,
                        'eval/std_score': std_score,
                        'eval/max_score': max_score,
                        'eval/min_score': min_score,
                        'eval/mean_len': sum(eval_len) / len(eval_len),
                    })

            time.sleep(1)


# ======================================================================================================================
# eval worker
# ======================================================================================================================
def start_eval_worker(agent, replay_buffer, storage, config):
    # start data worker
    eval_worker = EvalWorker.remote(agent, replay_buffer, storage, config)
    eval_worker.run.remote()
