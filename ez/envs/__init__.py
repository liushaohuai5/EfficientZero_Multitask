import os
import gymnasium
import humanoid_bench
import dmc2gym
# from gym.wrappers import Monitor
from .wrapper import *
import random
from dm_env import specs
from ez.utils.format import arr_to_str


def make_envs(game_setting, game_name, num_envs, seed, save_path=None, multi_task=False, difficulty=None, **kwargs):
    assert game_setting in ['Atari', 'DMC', 'Gym', 'Humanoid_Bench', 'ManiSkill']
    if game_setting == 'Atari':
        _env_fn = make_atari
    elif game_setting == 'Gym':
        _env_fn = make_gym
    elif game_setting == 'DMC':
        _env_fn = make_dmc
    elif game_setting == 'Humanoid_Bench':
        _env_fn = make_humanoid_bench
    elif game_setting == 'ManiSkill':
        _env_fn = make_maniskill
    else:
        raise NotImplementedError()

    # if game_setting == 'DMC':
    #     seed = random.randint(1, 1000)

    # seeds = random.sample(range(1000), num_envs)
    # seeds = [seed for _ in range(num_envs)]
    envs = [_env_fn(game_name,
                    seed=i + seed,
                    # seed=seeds[i],
                    save_path=save_path, **kwargs) for i in range(num_envs)]
    if multi_task:
        env_names = kwargs.get('env_names')
        envs = [_env_fn(env_name, seed=seed, save_path=save_path, **kwargs) for env_name in env_names]
    return envs


def make_env(game_setting, game_name, seed, save_path=None, **kwargs):
    assert game_setting in ['Atari', 'DMC', 'Gym', 'Humanoid_Bench', 'ManiSkill']
    if game_setting == 'Atari':
        _env_fn = make_atari
    elif game_setting == 'Gym':
        _env_fn = make_gym
    elif game_setting == 'DMC':
        _env_fn = make_dmc
    elif game_setting == 'Humanoid_Bench':
        _env_fn = make_humanoid_bench
    elif game_setting == 'ManiSkill':
        _env_fn = make_maniskill
    else:
        raise NotImplementedError()

    # seed = random.randint(1, 1000)

    env = _env_fn(game_name, seed=seed, save_path=save_path, **kwargs)
    return env


def make_atari(game_name, seed, save_path=None, **kwargs):
    from .atari import AtariWrapper
    """Make Atari games
    Parameters
    ----------
    game_name: str
        name of game (Such as Breakout, Pong)
    kwargs: dict
        skip: int
            frame skip
        obs_shape: (int, int)
            observation shape
        gray_scale: bool
            use gray observation or rgb observation
        seed: int
            seed of env
        max_episode_steps: int
            max moves for an episode
        save_path: str
            the path of saved videos; do not save video if None
            :param seed:
    """
    # params
    env_id = game_name + 'NoFrameskip-v4'
    gray_scale = kwargs.get('gray_scale')
    obs_to_string = kwargs.get('obs_to_string')
    skip = kwargs['n_skip'] if kwargs.get('n_skip') else 4
    obs_shape = kwargs['obs_shape'] if kwargs.get('obs_shape') else [3, 96, 96]
    max_episode_steps = kwargs['max_episode_steps'] if kwargs.get('max_episode_steps') else 108000 // skip
    episodic_life = kwargs.get('episodic_life')
    clip_reward = kwargs.get('clip_reward')

    env = gym.make(env_id)

    # set seed
    env.seed(seed)

    # random restart
    env = NoopResetEnv(env, noop_max=30)

    # frame skip
    env = MaxAndSkipEnv(env, skip=skip, image_based=True)

    # episodic trajectory
    if episodic_life:
        env = EpisodicLifeEnv(env)

    # reshape size and gray scale
    env = WarpFrame(env, width=obs_shape[1], height=obs_shape[2], grayscale=gray_scale)

    # set max limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # save video to given
    # if save_path:
    #     env = Monitor(env, directory=save_path, force=True)

    # your wrapper
    env = AtariWrapper(env, obs_to_string=obs_to_string, clip_reward=clip_reward)
    return env


def make_gym(game_name, seed, save_path=None, **kwargs):
    from .gym import GymWrapper
    save_path = kwargs.get('save_path')
    obs_to_string = kwargs.get('obs_to_string')
    skip = kwargs['n_skip'] if kwargs.get('n_skip') else 4

    env = gym.make(game_name)
    env = GymWrapper(env, obs_to_string=obs_to_string)

    # frame skip
    env = MaxAndSkipEnv(env, skip=skip, image_based=False)

    # set seed
    env.seed(seed)

    # save video to given
    # if save_path:
    #     env = Monitor(env, directory=save_path, force=True)

    # env = GymWrapper(env, obs_to_string=obs_to_string)
    return env


def make_humanoid_bench(game_name, seed, save_path=None, **kwargs):
    from .humanoid_bench import HumanoidBenchWrapper
    """Make Atari games
        Parameters
        ----------
        game_name: str
            name of game (Such as Breakout, Pong)
        kwargs: dict
            image_based: bool
                observation is image or state

        """
    # params

    # domain_name, task_name = game_name.split('_', 1)
    image_based = kwargs.get('image_based')
    # obs_shape = kwargs['obs_shape'] if kwargs.get('obs_shape') else [3, 96, 96]
    skip = kwargs['n_skip'] if kwargs.get('n_skip') else 2
    # save_path = kwargs.get('save_path')
    max_episode_steps = kwargs['max_episode_steps'] // skip
    clip_reward = kwargs.get('clip_reward')
    obs_to_string = kwargs.get('obs_to_string')
    render_mode = kwargs.get('render_mode')

    # # make env
    # if 'hand' not in game_name:
    game_name = game_name + '-v0'
    env = gymnasium.make(game_name, render_mode=render_mode, seed=seed)
    # your wrapper
    env = HumanoidBenchWrapper(env, obs_to_string=obs_to_string, clip_reward=clip_reward)

    # frame skip
    # env = MaxAndSkipEnv(env, skip=skip, image_based=image_based)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # save video to given
    # if save_path:
    #     env = Monitor(env, directory=save_path, force=True)
    return env


def make_maniskill(game_name, seed, save_path=None, **kwargs):
    import mani_skill.envs
    from .maniskill import ManiSkillWrapper#, ManiSkillRenderWrapper
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
    from mani_skill.utils.wrappers import ActionRepeatWrapper, FlattenActionSpaceWrapper, FlattenObservationWrapper

    image_based = kwargs.get('image_based')
    skip = kwargs['n_skip'] if kwargs.get('n_skip') else 2
    # save_path = kwargs.get('save_path')
    max_episode_steps = kwargs['max_episode_steps'] // skip
    # clip_reward = kwargs.get('clip_reward')
    # obs_to_string = kwargs.get('obs_to_string')
    # render_mode = kwargs.get('render_mode')

    env = gymnasium.make(f'{game_name}-v1', num_envs=1, reconfiguration_freq=1)
    env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)

    # env = ActionRepeatWrapper(env, repeat=skip)
    # print('obs_shape', env.observation_space)
    # print('act_shape', env.action_space)
    # print('ctrl_mode', env.control_mode)
    # env.reset()
    # obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    # import ipdb
    # ipdb.set_trace()

    # env = FlattenActionSpaceWrapper(env)
    # env = FlattenObservationWrapper(env)
    env = ManiSkillWrapper(env, action_repeat=skip)
    # env = MaxAndSkipEnv(env, skip=skip, image_based=image_based)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    # env = ManiSkillRenderWrapper(env)
    return env


def make_dmc(game_name, seed, save_path=None, **kwargs):
    from .dmc import DMCWrapper
    """Make Atari games
    Parameters
    ----------
    game_name: str
        name of game (Such as Breakout, Pong)
    kwargs: dict
        image_based: bool
            observation is image or state

    """
    # params
    if 'CMU' in game_name:
        domain_name, task_name = game_name.rsplit('_', 1)
    else:
        domain_name, task_name = game_name.split('_', 1)
    image_based = kwargs.get('image_based')
    obs_shape = kwargs['obs_shape'] if kwargs.get('obs_shape') else [3, 96, 96]
    skip = kwargs['n_skip'] if kwargs.get('n_skip') else 2
    # save_path = kwargs.get('save_path')
    # gray_scale = kwargs.get('gray_scale')
    max_episode_steps = kwargs['max_episode_steps'] // skip
    if 'finger' in domain_name:
        max_episode_steps = 200
    clip_reward = kwargs.get('clip_reward')
    obs_to_string = kwargs.get('obs_to_string')
    camera_id = 2 if 'quadruped' in domain_name else 0

    # # make env
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        visualize_reward=False,
        from_pixels=image_based,
        height=obs_shape[1] if image_based else 96,
        width=obs_shape[1] if image_based else 96,
        frame_skip=skip,
        channels_first=False,
        camera_id=camera_id,
        # time_limit=max_episode_steps,
    )

    # env = MaxAndSkipEnv(env, skip=skip, image_based=image_based)

    env = DMCWrapper(env, obs_to_string=obs_to_string, clip_reward=clip_reward)

    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # save video to given
    # if save_path:
    #     env = Monitor(env, directory=save_path, force=True)

    # your wrapper
    return env