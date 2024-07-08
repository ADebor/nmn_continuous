import numpy as np

import pygame

pygame.init()

import gymnasium as gym
from gymnasium import spaces


class MovingTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, max_steps=50, margin=1.0, offset_mult=0):
        self._max_steps = max_steps
        self._max_bound = 10
        self._margin = margin
        self._offset_mult = offset_mult

        bound = np.array([0.5 * self._max_bound])
        self.observation_space = spaces.Box(low=-bound, high=bound, dtype=np.float64)

        self.action_space = spaces.Box(
            low=-2 * self._max_bound,
            high=2 * self._max_bound,
            shape=(1,),
            dtype=np.float64,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.resolution = (640, 480)
        self.window = None
        self.pygame_events = []
        self.font = pygame.font.Font("freesansbold.ttf", 15)
        self.clock = None

    def _rand_range(self, max_range, size):
        return np.random.uniform(-max_range, max_range, size)

    def _get_obs(self):
        return self._initial_pose

    def reset(self, seed=None, options=None):
        self._episode_step = 0

        self._last_action = np.array([0])

        self._initial_pose = self._rand_range(0.5 * self._max_bound, 1)
        self._offset = self._rand_range(self._max_bound, 1) * self._offset_mult

        obs = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return obs, {}

    def step(self, action):
        self._episode_step += 1
        self._last_action = np.clip(action, -2 * self._max_bound, 2 * self._max_bound)

        goal = self._initial_pose - self._offset

        dist = np.abs(goal - self._last_action).item()

        if dist < self._margin:
            reward = 10
            self._initial_pose = self._rand_range(0.5 * self._max_bound, 1)
        else:
            reward = np.clip(-dist, -2 * self._max_bound, 2 * self._max_bound)

        obs = self._get_obs()
        terminated = self._episode_step >= self._max_steps

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, False, {}

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.resolution[0], self.resolution[1])
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.resolution[0], self.resolution[1]))
        canvas.fill((255, 255, 255))

        y_center = self.resolution[1] / 2

        val_to_screen = lambda x: int(
            (x + 2 * self._max_bound) / (4 * self._max_bound) * self.resolution[0]
        )

        # Draw base line
        pygame.draw.line(
            canvas, (0, 0, 0), (0, y_center), (self.resolution[0], y_center), 2
        )

        # Draw line showing position 0
        pygame.draw.line(
            canvas,
            (0, 0, 0),
            (self.resolution[0] / 2, y_center - 5),
            (self.resolution[0] / 2, y_center + 5),
            2,
        )

        # Draw initial position
        target_x = val_to_screen(self._initial_pose.item())
        pygame.draw.line(
            canvas,
            (0, 0, 0),
            (target_x, y_center - 20),
            (target_x, y_center + 20),
            2,
        )

        # Draw target
        goal = (self._initial_pose - self._offset).item()
        offset_x = val_to_screen(goal)
        pygame.draw.line(
            canvas,
            (0, 255, 255),
            (offset_x, y_center - 20),
            (offset_x, y_center + 20),
            2,
        )

        # Draw target margins
        margin_x = val_to_screen(goal + self._margin)
        margin_x_ = val_to_screen(goal - self._margin)
        for x in [margin_x, margin_x_]:
            pygame.draw.line(
                canvas,
                (100, 100, 100),
                (x, y_center - 20),
                (x, y_center + 20),
                2,
            )

        # Draw action
        action_x = val_to_screen(self._last_action.item())
        pygame.draw.line(
            canvas,
            (0, 255, 0),
            (action_x, y_center - 20),
            (action_x, y_center + 20),
            2,
        )

        text = self.font.render(
            "Offset: {}".format(self._offset),
            True,
            (255, 0, 0),
            None,
        )
        canvas.blit(text, (0, 0))

        text = self.font.render(
            "Action: {}".format(self._last_action),
            True,
            (255, 0, 0),
            None,
        )
        canvas.blit(text, (0, 15))

        if self.render_mode != "human":  # rgb_array
            return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))

        self.pygame_events = pygame.event.get()
        self.window.blit(canvas, canvas.get_rect())

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
