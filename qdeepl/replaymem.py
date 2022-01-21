from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], capacity)

    def push(self, current_state_frame_stack, action, reward, next_state_frame_stack, done):
        self.memory.append(Transition(current_state_frame_stack, action, reward, next_state_frame_stack, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)