import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

class DQNAgent:
    def __init__(
            self, actions_space=None, frame_stack_num=3, mem_size=5000, gamma=0.95, ep=1.0, ep_min=0.1, ep_decay=0.9999,
            lr=0.0001
    ):
        if actions_space is None:
            self.actions_space = [
                (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),     # Action Space Structure
                (-1, 1, 0), (0, 1, 0), (1, 1, 0),           # (Steering Wheel, Gas, Break)
                (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),     #  Range -1~1      0~1   0~1
                (-1, 0, 0), (0, 0, 0), (1, 0, 0)
            ]
        else:
            self.actions_space = actions_space

        self.frame_stack_num = frame_stack_num
        self.memory = deque(maxlen=mem_size)
        self.gamma = gamma
        self.ep = ep
        self.ep_min = ep_min
        self.ep_decay = ep_decay
        self.lr = lr

        self.model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu',
                         input_shape=(96, 96, self.frame_stack_num)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(len(self.actions_space), activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.lr, epsilon=1e-7))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, self.actions_space.index(action), reward, next_state, done))

    def act(self, state):
        if np.random.rand() > self.ep:
            act_values = self.model.predict(np.expand_dims(state, axis=0))
            action_index = np.argmax(act_values[0])
        else:
            action_index = random.randrange(len(self.actions_space))
        return self.actions_space[action_index]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        train_state = []
        train_target = []
        for state, action_index, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0))[0]
            if done:
                target[action_index] = reward
            else:
                t = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
                target[action_index] = reward + self.gamma * np.amax(t)

            train_state.append(state)
            train_target.append(target)
        self.model.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0)
        if self.ep > self.ep_min:
            self.ep *= self.ep_decay

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.target_model.save_weights(name)