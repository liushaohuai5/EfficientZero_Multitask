import os
import sys
sys.path.append(os.getcwd())

import time
import torch
import ray
import copy
import cv2
import hydra
import multiprocessing
import numpy as np
import imageio
from PIL import Image, ImageDraw

from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from torch.cuda.amp import autocast as autocast
from ez.utils.format import formalize_obs_lst
from ez.mcts.cy_mcts import Gumbel_MCTS

@torch.no_grad()
def eval_pure(agent, model, n_episodes, save_path, config, envs, max_steps=None, use_pb=False, verbose=0, task_idxs=None):
    model.cuda()
    model.eval()

    # prepare logs
    if save_path is not None:
        video_path = save_path / 'recordings'
        video_path.mkdir(parents=True, exist_ok=True)
    else:
        video_path = None

    # make env
    if max_steps is not None:
        config.env.max_episode_steps = max_steps
    n_episodes = len(envs)

    dones = np.array([False for _ in range(n_episodes)])
    if use_pb:
        pb = tqdm(np.arange(max_steps), leave=True)
    ep_ori_rewards = np.zeros(n_episodes)
    ep_ori_lens = [0 for _ in range(n_episodes)]

    # initialization
    stack_obs_windows, game_trajs = agent.init_envs(envs, max_steps, seed=config.env.base_seed)

    # set infinity trajectory size
    [traj.set_inf_len() for traj in game_trajs]

    # begin to evaluate
    step = 0
    frames = [[] for _ in range(n_episodes)]
    rewards = [[] for _ in range(n_episodes)]
    infos = {}
    while not dones.all():
        # debug
        if verbose:
            import ipdb
            ipdb.set_trace()

        # stack obs
        current_stacked_obs = formalize_obs_lst(stack_obs_windows, image_based=config.env.image_based)
        # obtain the statistics at current steps
        with torch.no_grad():
            with autocast():
                states, values, policies, _ = model.initial_inference(current_stacked_obs,
                                                                      task_idxs=torch.from_numpy(task_idxs).long().cuda())

        values = values.detach().cpu().numpy().flatten()

        if config.env.continuous_action:
            r_values, r_policies, best_actions, _, sampled_actions, _, _ = \
                Gumbel_MCTS(config).run_multi_continuous(
                    model, n_episodes, states, values, policies, add_noise=False, use_gumbel_noise=False, task_idxs=task_idxs
                )
        else:
            r_values, r_policies, best_actions, _, _, _ = \
                Gumbel_MCTS(config).run_multi_discrete(
                    model, n_episodes, states, values, policies, use_gumbel_noise=False, task_idxs=task_idxs
                )

        # step action in environments
        for i in range(n_episodes):
            task_idx = int(task_idxs[i])
            task_name = config.env.env_names[task_idx] if config.env.multi_task else config.env.game

            if dones[i]:
                continue

            action = best_actions[i]
            if config.env.multi_task:
                if config.env.env == 'Atari':
                    assert action < int(sum(config.env.action_masks[task_idx])), f'action out of range'
                if config.env.continuous_action:
                    action = action[:int(sum(config.env.action_masks[task_idx]))]
            obs, reward, done, info = envs[i].step(action)
            if config.env.multi_task:
                if not config.env.image_based:
                    obs = np.concatenate([obs, np.zeros(config.env.obs_shape - obs.shape[0])])  # pad to same dim with 0
            if config.env.image_based:
                pic = obs
            else:
                pic = envs[i].render(mode='rgb_array')

            frames[i].append(pic)
            rewards[i].append(info['raw_reward'])
            dones[i] = done

            # save data to trajectory buffer
            game_trajs[i].store_search_results(values[i], r_values[i], r_policies[i])
            game_trajs[i].append(action, obs, reward)
            if config.env.env == 'Atari':
                game_trajs[i].snapshot_lst.append(envs[i].ale.cloneState())
            elif config.env.env == 'DMC':
                game_trajs[i].snapshot_lst.append(envs[i].physics.get_state())

            del stack_obs_windows[i][0]
            stack_obs_windows[i].append(obs)

            # log
            ep_ori_rewards[i] += info['raw_reward']
            ep_ori_lens[i] += 1

            if dones[i]:
                if config.env.env == 'ManiSkill':
                    print(task_name, info['episode'])
                    if infos.get(task_name) is None:
                        infos[task_name] = {}
                        infos[task_name]['success_once'] = 0
                        infos[task_name]['success_at_end'] = 0
                        infos[task_name]['success_once'] += int(info['episode']['success_once'])
                        infos[task_name]['success_at_end'] += int(info['episode']['success_at_end'])
                    else:
                        infos[task_name]['success_once'] += int(info['episode']['success_once'])
                        infos[task_name]['success_at_end'] += int(info['episode']['success_at_end'])

        step += 1
        if use_pb:
            pb.set_description('{} In step {}, take action {}, scores: {}(max: {}, min: {}) currently.'
                               ''.format(config.env.game, step, best_actions,
                                         ep_ori_rewards.mean(), ep_ori_rewards.max(), ep_ori_rewards.min()))
            pb.update(1)

    for i, env in enumerate(envs):
        task_idx = int(task_idxs[i])
        if config.env.multi_task:
            writer = imageio.get_writer(video_path / f'{config.env.env_names[task_idx]}_epi_{i%config.env.task_num}_{max_steps}.mp4')
        else:
            writer = imageio.get_writer(video_path / f'epi_{i}_{max_steps}.mp4')
        rewards[i][0] = sum(rewards[i])

        j = 0
        for frame, reward in zip(frames[i], rewards[i]):
            frame = Image.fromarray(frame)
            frame = np.array(frame)
            writer.append_data(frame)
            j += 1
        writer.close()

    return ep_ori_rewards, ep_ori_lens, infos
