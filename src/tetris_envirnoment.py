import gymnasium as gym
import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
import numpy as np


class TetrisEnv(gym.Env):
    def __init__(self):
        env = gym_tetris.make('TetrisA-v0')
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.action_space = self.env.action_space 
        self.observation_space = self.env.observation_space
        n_rows, n_cols, z = self.observation_space.shape
        self.n_observation_states = n_rows * n_cols * z
        self.state_shape = (n_rows, n_cols, z)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.array(obs).astype(np.float32)
        return obs, reward, done, info

    def render(self, mode='human'):
        self.env.render(mode=mode)

    def close(self):
        self.env.close()


