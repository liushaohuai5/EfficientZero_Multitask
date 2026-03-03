from ..base import BaseWrapper


class ManiSkillWrapper(BaseWrapper):
    """
    Make your own wrapper: ManiSkill Wrapper
    """
    def __init__(self, env, obs_to_string=False, clip_reward=False, action_repeat=1):
        super().__init__(env, obs_to_string, clip_reward)
        self.length = 0
        self.action_repeat = action_repeat

    def reset(self, seed=None):
        self.length = 0
        obs, info = self.env.reset(seed=seed)
        return obs

    def step(self, action):
        total_reward = 0
        terminated = truncated = False
        info = {}
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        self.length += 1
        done = terminated | truncated
        info['raw_reward'] = total_reward
        return obs.squeeze(), total_reward, done, info

    def render(self, mode='rgb_array', **kwargs):
        return self.env.render_rgb_array().squeeze().detach().cpu().numpy()


# class ManiSkillRenderWrapper(BaseWrapper):
#
#     def __init__(self, env, obs_to_string=False, clip_reward=False):
#         super().__init__(env, obs_to_string, clip_reward)
#
#     def render(self, mode='rgb_array', **kwargs):
#         return self.env.render_rgb_array().squeeze().detach().cpu().numpy()

