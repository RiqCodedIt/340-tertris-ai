import numpy as np
import tensorflow as tf
import gymnasium as gym
import gym_tetris

class Agent:
    def __init__(self, tetris, alpha=0.1, gamma=0.9, epsilon=0.1, hidden_layers=[32, 32]):
        self.state_space = tetris.observation_space
        self.action_space = tetris.action_space
        self.n_states = tetris.n_observation_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.hidden_layers = hidden_layers      
        self.q_table = np.zeros((self.n_states, self.action_space.n)) 
        #self.model = self._build_model()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(self.n_states,), activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_space.n, activation='linear')
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
        
    def _build_model(self):
        inputs = tf.keras.layers.Input(shape=self.state_space.reshape(1,-1))
        x = tf.keras.layers.Dense(32, activation='relu')(inputs)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.action_space.n)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.action_space.n)
        else:
            q_values = self.model.predict(np.array([state]).reshape(1,-1))[0]
            action = np.argmax(q_values)
        return action
        
    
    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[self.state_space.index(state)][self.action_space.index(action)]
        next_q = np.max(self.q_table[self.state_space.index(next_state)])
        target_q = reward + self.gamma * next_q
        updated_q = (1 - self.alpha) * current_q + self.alpha * target_q
        self.q_table[self.state_space.index(state)][self.action_space.index(action)] = updated_q
        
    def update_model(self, states, actions, targets):
        X = np.array(states)
        Y = self.model.predict(X)
        for i in range(len(states)):
            Y[i][self.action_space.index(actions[i])] = targets[i]
        self.model.fit(X, Y, epochs=1, verbose=0)
        
    def train(self, state, action, reward, next_state, done): #was 1000
        q_value = self.q_table[state, action]
        next_q_value = np.max(self.q_table[next_state])
        target = reward + self.gamma * next_q_value * (1 - done)
        self.q_table[state, action] = (1 - self.alpha) * q_value + self.alpha * target

        # Train the TensorFlow model using experience replay
        self.model.fit(np.array([state]).reshape(1,-1), np.array([self.q_table[state]]), verbose=0)
       
""" env = tetris.env
        for episode in range(num_episodes):
            print("This is the ", episode, " episode")
            state = env.reset()
            done = False
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                env.render(mode='human')

                self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]))
                #self.update_q_table(state, action, reward, next_state)
                state = next_state
        np.save('q_table.npy', self.q_table)"""


            #states, actions, rewards, next_states = env.memory
            #targets = []
            #for i in range(len(states)):
                #next_q = np.max(self.model.predict(np.array([next_states[i]]))[0])
                #target_q = rewards[i] + self.gamma * next_q
                #targets.append(target_q)
            #self.update_model(states, actions, targets)
        