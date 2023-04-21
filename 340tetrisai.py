import numpy as np
import tensorflow as tf

class TetrisQLearner:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.9, epsilon=0.1, hidden_layers=[32, 32]):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.hidden_layers = hidden_layers
        
        self.q_table = np.zeros((len(state_space), len(action_space)))
        self.model = self._build_model()
        
    def _build_model(self):
        input_shape = (len(self.state_space),)
        output_shape = (len(self.action_space),)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(self.hidden_layers[0], input_shape=input_shape, activation='relu'))
        for i in range(1, len(self.hidden_layers)):
            model.add(tf.keras.layers.Dense(self.hidden_layers[i], activation='relu'))
        model.add(tf.keras.layers.Dense(len(self.action_space), activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.alpha))
        return model
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            q_values = self.model.predict(np.array([state]))[0]
            action = self.action_space[np.argmax(q_values)]
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
        
    def train(self, env, num_episodes=1000):
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.get_action(state)
                next_state, reward, done = env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
            states, actions, rewards, next_states = env.memory
            targets = []
            for i in range(len(states)):
                next_q = np.max(self.model.predict(np.array([next_states[i]]))[0])
                target_q = rewards[i] + self.gamma * next_q
                targets.append(target_q)
            self.update_model(states, actions, targets)