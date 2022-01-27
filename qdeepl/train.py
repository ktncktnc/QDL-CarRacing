import argparse
import gym
from collections import deque
from model import DQNAgent
from funcs import *

RENDER = False
STARTING_EPISODE = 1
ENDING_EPISODE = 500
SKIP_FRAMES = 2
TRAINING_BATCH_SIZE = 64
SAVE_TRAINING_FREQUENCY = 5
UPDATE_TARGET_MODEL_FREQUENCY = 10

if __name__ == '__main__':
    disable_view_window()
    env = gym.make("CarRacing-v0")

    agent = DQNAgent()

    for e in range(STARTING_EPISODE, ENDING_EPISODE):
        print("Episode " + str(e))
        init_state = env.reset()
        init_state = process_state_image(init_state)

        total_reward = 0
        negative_reward_counter = 0
        state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        done = False

        while True:
            if RENDER:
                env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)

            reward = 0
            for _ in range(SKIP_FRAMES + 1):
                next_state, r, done, info = env.step(action)
                reward += r
                if done:
                    break

            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

            # Boost reward if car run fast
            if action[1] == 1 and action[2] == 0:
                reward *= 1.5

            total_reward += reward

            next_state = process_state_image(next_state)

            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            if done or negative_reward_counter >= 25 or total_reward < 0:
                print(
                    'Episode: {}/{}, Scores(Time Frames): {}, Total Rewards(adjusted): {:.2}, Epsilon: {:.2}'.format(e,
                        ENDING_EPISODE, time_frame_counter, float(total_reward), float(agent.ep)))
                break

            if len(agent.memory) > TRAINING_BATCH_SIZE:
                agent.replay(TRAINING_BATCH_SIZE)

            time_frame_counter += 1
            if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
                agent.update_target_model()

            if e % SAVE_TRAINING_FREQUENCY == 0:
                agent.save('./save/trial_{}.h5'.format(e))

        env.close()

    print("Ok")
