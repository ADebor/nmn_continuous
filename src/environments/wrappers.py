import gymnasium as gym
import numpy as np
import pygame


class MetaObs(gym.Wrapper):
    def __init__(self, env, observation_type: str = "full"):
        super().__init__(env)
        init_obs_dim = self.observation_space.shape[0]
        self.obs_cfg = observation_type

        if observation_type == "full" or observation_type == "full_current":
            self.observation_space = gym.spaces.Box(
                low=np.concatenate(
                    [
                        self.observation_space.low,
                        self.observation_space.low,
                        self.action_space.low,
                        -np.inf * np.ones(1),
                    ]
                ),
                high=np.concatenate(
                    [
                        self.observation_space.high,
                        self.observation_space.high,
                        self.action_space.high,
                        np.inf * np.ones(1),
                    ]
                ),
                dtype=np.float32,
            )
        elif observation_type == "current":
            self.observation_space = gym.spaces.Box(
                low=np.concatenate(
                    [
                        self.observation_space.low,
                        self.action_space.low,
                        -np.inf * np.ones(1),
                    ]
                ),
                high=np.concatenate(
                    [
                        self.observation_space.high,
                        self.action_space.high,
                        np.inf * np.ones(1),
                    ]
                ),
                dtype=np.float32,
            )

        # keep track of the initial observation dimension
        self.observation_space.init_obs_dim = init_obs_dim
        self.prev_obs = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        prev_obs = self.prev_obs
        self.prev_obs = obs

        if self.obs_cfg == "full":
            obs_list = [obs, prev_obs, action, [reward]]
        elif self.obs_cfg == "full_current":
            obs_list = [obs, obs, action, [reward]]
        elif self.obs_cfg == "current":
            obs_list = [obs, action, [reward]]

        return (
            np.concatenate(obs_list, dtype=np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        if self.obs_cfg == "full":
            obs_list = [
                obs,
                np.zeros_like(obs),
                np.zeros(shape=self.action_space.shape[0]),
                [0.0],
            ]
        elif self.obs_cfg == "full_current":
            obs_list = [
                obs,
                obs,
                np.zeros(shape=self.action_space.shape[0]),
                [0.0],
            ]
        elif self.obs_cfg == "current":
            obs_list = [
                obs,
                np.zeros(shape=self.action_space.shape[0]),
                [0.0],
            ]

        obs = np.concatenate(
            obs_list,
            dtype=np.float32,
        )
        return obs, info


class RenderActionWrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        self._last_action = None
        self.font = pygame.font.Font("freesansbold.ttf", 15)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_action = action
        return obs, reward, terminated, truncated, info

    def render(self):
        _ = self.env.render()
        text = self.font.render(
            "Actions: {}".format(self._last_action[0]),
            True,
            (255, 0, 0),
            None,
        )
        self.env.unwrapped.screen.blit(text, (0, 0))
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.env.unwrapped.screen)),
            axes=(1, 0, 2),
        )


class RewardAsInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["reward"] = reward
        return obs, reward, terminated, truncated, info
