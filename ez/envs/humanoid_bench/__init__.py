from ..base import BaseWrapper
import cv2
import numpy as np
import copy


class HumanoidBenchWrapper(BaseWrapper):
    """
    Make your own wrapper: Humanoid_Bench Wrapper
    """
    def __init__(self, env, obs_to_string=False, clip_reward=False):
        super().__init__(env, obs_to_string, clip_reward)
        self.action_t = [np.zeros(env.action_space.shape) for _ in range(2)]

    def step(self, action):
        obs, reward, _, done, info = self.env.step(action)
        info['raw_reward'] = reward
        self.action_t.append(action)
        del self.action_t[0]
        return obs, reward, done, info

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs

    def render(self, mode='rgb_array', **kwargs):
        return self.env.render()

# class HumanoidBenchWrapper(BaseWrapper):
#     """
#     Make your own wrapper: DMC Wrapper
#     """
#     def __init__(self, env, obs_to_string=False, clip_reward=False):
#         super().__init__(env, obs_to_string, clip_reward)


