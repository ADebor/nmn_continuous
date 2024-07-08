import numpy as np

import pygame

import gymnasium as gym


class MultipleReferencesEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
        self,
        render_mode=None,
        max_steps=200,
        max_reward=100,
        min_reward=-50,
        *args,
        **kwargs,
    ):
        self.max_steps = max_steps
        self._target_rewards = [min_reward, max_reward]
        self._area = 5

        self._agent_spawn_area = np.array([[1.0, 1.0], [4.0, 4.0]])
        self._agent_move_speed = 0.25

        self._target_spawn_area = np.array([[1.5, 1.5], [3.5, 3.5]])
        self._target_radius = 0.4

        bound = np.array([np.inf for _ in range(len(self._target_rewards) * 2)])
        self.observation_space = gym.spaces.Box(
            low=-bound, high=bound, dtype=np.float64
        )

        self.action_space = gym.spaces.Box(
            low=-np.pi,
            high=np.pi,
            shape=(1,),
            dtype=np.float64,
        )
        self.lowest_reward = min(self._target_rewards)
        self.highest_reward = max(self._target_rewards)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.resolution = (640, 640)
        self.window = None
        self.clock = None
        self.font = pygame.font.Font("freesansbold.ttf", 15)

    def _get_observation(self):
        dist = []
        for target_pos in self._targets_pos:
            dist.append(target_pos - self._agent_pos)
        return np.array(dist).flatten()

    def _move(self, pos, action):
        return pos + self._agent_move_speed * np.array([np.cos(action), np.sin(action)])

    @property
    def alpha(self):
        return int(self._target_rewards[0] > self._target_rewards[1])

    def reset(self, seed=None, options=None):
        self._steps = 0

        self._last_action = None

        np.random.shuffle(self._target_rewards)

        self._agent_pos = np.random.uniform(
            self._agent_spawn_area[0], self._agent_spawn_area[1]
        )

        # Generate target positions that don't overlap
        self._targets_pos = []
        for _ in range(len(self._target_rewards)):
            overlaps = True
            while overlaps:
                overlaps = False
                pos = np.random.uniform(
                    self._target_spawn_area[0], self._target_spawn_area[1]
                )
                for target_pos in self._targets_pos:
                    if np.linalg.norm(pos - target_pos) <= 2 * self._target_radius:
                        overlaps = True
                        break
                else:
                    if np.linalg.norm(pos - self._agent_pos) <= self._target_radius:
                        overlaps = True
            self._targets_pos.append(pos)

        self._current_trajectory = [self._agent_pos.copy()]
        self._trajectories = []

        if self.render_mode == "human":
            self._render_frame()

        return self._get_observation(), {}

    def step(self, action):
        self._steps += 1

        self._last_action = action.item()

        self._agent_pos = self._move(self._agent_pos, self._last_action) % self._area

        self._current_trajectory.append(self._last_action)

        reward = 0
        for target_pos, target_reward in zip(self._targets_pos, self._target_rewards):
            # Target reached, resample agent position
            if np.linalg.norm(target_pos - self._agent_pos) <= self._target_radius:
                reward = target_reward
                overlaps = True
                while overlaps:
                    overlaps = False
                    self._agent_pos = np.random.uniform(
                        self._agent_spawn_area[0], self._agent_spawn_area[1]
                    )
                    for target_pos in self._targets_pos:
                        if (
                            np.linalg.norm(self._agent_pos - target_pos)
                            <= self._target_radius
                        ):
                            overlaps = True
                            break
                self._trajectories.append(self._current_trajectory)
                self._current_trajectory = [self._agent_pos.copy()]

        done = self._steps >= self.max_steps

        if self.render_mode == "human":
            self._render_frame()

        return self._get_observation(), reward, done, False, {}

    def _to_screen(self, coord):
        x, y = coord
        return np.array(
            [
                int(x * self.resolution[0] / self._area),
                self.resolution[1] - int(y * self.resolution[1] / self._area),
            ]
        )

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

        for traj in self._trajectories:
            pos = traj[0]
            for action in traj[1:]:
                next_pos = self._move(pos, action)
                pygame.draw.line(
                    canvas,
                    (153, 192, 255),
                    self._to_screen(pos),
                    self._to_screen(next_pos),
                    width=3,
                )
                pos = next_pos % self._area

        pos = self._current_trajectory[0]
        for action in self._current_trajectory[1:]:
            next_pos = self._move(pos, action)
            pygame.draw.line(
                canvas,
                (0, 98, 255),
                self._to_screen(pos),
                self._to_screen(next_pos),
                width=3,
            )
            pos = next_pos % self._area

        for target_pos, r in zip(self._targets_pos, self._target_rewards):
            if r < 0:
                color = (255 * r / self.lowest_reward, 0, 0)
            else:
                color = (0, 255 * r / self.highest_reward, 0)
            pygame.draw.circle(
                canvas,
                color,
                self._to_screen(target_pos),
                int((self._target_radius / 4.0) * self.resolution[0] / self._area),
            )
            pygame.draw.circle(
                canvas,
                color,
                self._to_screen(target_pos),
                int(self._target_radius * self.resolution[0] / self._area),
                width=4,
            )

        pygame.draw.circle(
            canvas,
            (0, 0, 0),
            self._to_screen(self._agent_pos),
            5,
        )

        if self._last_action is not None:
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                self._to_screen(self._agent_pos),
                self._to_screen(
                    self._agent_pos
                    + 0.5
                    * np.array([np.cos(self._last_action), np.sin(self._last_action)])
                ),
                width=3,
            )

        text = self.font.render(
            f"locations: {self._targets_pos}",
            True,
            (0, 0, 0),
            (153, 192, 255),
        )
        canvas.blit(text, (0, self.resolution[1] - 35))

        text = self.font.render(
            f"rewards: {self._target_rewards}",
            True,
            (0, 0, 0),
            (153, 192, 255),
        )
        canvas.blit(text, (0, self.resolution[1] - 15))

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
