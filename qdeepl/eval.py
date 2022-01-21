import gym
import argparse
from model import DQNAgent
from funcs import  *
from collections import deque

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a DQN agent to play CarRacing.')
    parser.add_argument("-m", '--model', help="Path to trained model")

    args = parser.parse_args()
    env = gym.make("CarRacing-v0")

    agent = DQNAgent()
    agent.load(args.model)
    agent.ep = 0

    init_state = env.reset()
    init_state = process_state_image(init_state)
    state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)

    while True:
        env.render()

        current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
        action = agent.act(current_state_frame_stack)

        next_state, reward, done, _ = env.step(action)
        next_state = process_state_image(next_state)
        state_frame_stack_queue.append(next_state)

        if done:
            break

    env.close()



