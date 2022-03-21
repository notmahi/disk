"""
Simple square environment where the agent can run around however it wants.
"""

import numpy as np

import gym
from gym import spaces

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SquareEnv(gym.Env):
    metadata = {
        "render.modes": ["rgb_array"],
    }

    def __init__(
        self, size, episode_length, temperature=-1.0, enable_render=True, alpha_viz=0.01
    ):
        self.enable_render = enable_render
        self.size = size
        self._max_episode_steps = episode_length
        self.episode_length = episode_length

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-self.size, high=self.size, shape=(2,), dtype=np.float32
        )

        self.temperature = temperature
        # Initial conditions
        self.state = None
        self.movement_memory = np.empty(shape=(episode_length + 1, 2))
        self.average_movement = np.zeros_like(self.movement_memory)
        self._current_step = 0
        self.fig = None
        self.alpha_viz = alpha_viz

        self.skill_now = 0
        self.avg_skill_paths = []

    def step(self, action):
        self.action = action / (np.linalg.norm(action) + 1e-9)
        if self.temperature != -1:
            self.state += np.tanh(action * self.temperature)
        else:
            self.state += action
        self.action = action
        self.state = np.clip(
            self.state, self.observation_space.low, self.observation_space.high
        )
        self._current_step += 1
        self.movement_memory[self._current_step] = self.state
        reward = 0
        done = self._current_step >= self.episode_length
        info = {}

        return self.state.copy(), reward, done, info

    def reset(self):
        # Start at the zero state.
        self.state = np.zeros(2)
        self._current_step = 0
        # And reset the movement phase
        self.average_movement = (
            self.alpha_viz * self.movement_memory
            + (1 - self.alpha_viz) * self.average_movement
        )
        self.movement_memory[self._current_step] = self.state

        # Rendering details
        if self.fig is not None:
            plt.close()
        fig = plt.figure()
        self._inch_size = 2.56
        fig.set_size_inches((self._inch_size, self._inch_size))
        plt.axis("off")
        plt.xlim([-self.size, self.size])
        plt.ylim([-self.size, self.size])
        self.fig = fig
        self.plt = plt
        self.line_plot = self.plt.plot(self.movement_memory[:1])[0]
        # Now plot all the older averages too
        for avg in self.avg_skill_paths:
            self.avg_plot = self.plt.plot(avg[:, 0], avg[:, 1], alpha=0.2)
        self.fig.canvas.draw()

        return self.state.copy()

    def render(self, height, width, camera_id, mode="rgb_array"):
        if mode == "rgb_array":
            self.fig.set_dpi(height / self._inch_size)
            self.line_plot.set_data(
                self.movement_memory[: self._current_step, 0],
                self.movement_memory[: self._current_step, 1],
            )
            self.plt.title(str(self.action))
            self.fig.canvas.draw()
            # this rasterizes the figure
            return np.array(self.fig.canvas.renderer._renderer)
        else:

            return None

    def add_new_skill(self):
        self.avg_skill_paths.append(self.average_movement)
        self.average_movement = np.zeros_like(self.movement_memory)
