import gym
import gym_tetris
import numpy as np


class TetrisEnv(gym.Env):
    def __init__(self):
        self.env = gym_tetris.make('TetrisA-v0')
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(20, 10))

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


env = TetrisEnv()
obs = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    
env.close()