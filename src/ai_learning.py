import gymnasium as gym 
import gym_tetris
import numpy as np
from tetris_ai import Agent
from tetris_envirnoment import TetrisEnv


# Initialize the AI agent and Tetris envirnoment
tetris = TetrisEnv()
agent = Agent(tetris)
env = tetris.env
# Set the hyperparameters

n_episodes = 10000

# Run the Q-learning algorithm
for i in range(n_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Select an action using the AI agent
        action = agent.get_action(state)
        
        next_state, reward, done, _ = env.step(action)

        env.render(mode='human')

        # Update the AI agent's Q-table and model
        agent.train(state, action, reward, next_state, done)
        
        state = next_state

np.save('q_table.npy', agent.q_table)   
env.close()



