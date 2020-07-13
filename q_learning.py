import os
import time

import matplotlib.pyplot as plt
import numpy as np
import gym

from visualize_model import draw_matrix

class HyperParams:
    def __init__(self, learning_rate, discount_factor):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor


def get_env():
    env = gym.make("MountainCar-v0")
    return env


def map_observation_to_state(observation_ranges, observation_dims):
    def mapper(observation_values):
        return *(pos_in_range(start, stop, dim, value) 
            for (start, stop), value, dim in zip(observation_ranges, observation_values, observation_dims)),

    return mapper


def pos_in_range(start, stop, slice_num, value):
    return int((value - start) / (stop - start) * slice_num)


def action_from_observation(observation, q_table, os_mapper):
    state = os_mapper(observation)
    # print(f"state: {state}")
    action = np.argmax(q_table[state])

    return action


def learn_from_reward(
        reward_t, 
        q_table, 
        state_t, 
        state_tp1, 
        action_t, 
        hyper_params:HyperParams):
    old_value = q_table[state_t][action_t]
    estimate_of_optimal_future_value = \
        np.max(q_table[state_tp1])

    # update q_table
    q_table[state_t][action_t] = \
        old_value + hyper_params.learning_rate * (
            reward_t + hyper_params.discount_factor * \
                estimate_of_optimal_future_value - old_value
        )


def calculate_reward(prev_position, position):
    if position >= 0.5:
        return 100
    # elif position > prev_position:
    #     return (position - prev_position)*1000
    else:
        return -1
    

def norm_array(input_array, shift=True):
    maxval = max(input_array)
    minval = min(input_array)
    if shift:
        normed = [(xx - minval) / (maxval - minval) for xx in input_array]
    else:
        normed = [(xx - minval) / (maxval - minval) for xx in input_array]
    return normed


def save_q_table(q_table, model_id):
    # write q_table
    output_path = f"./models/q_table-{model_id}.npy"
    np.save(output_path, q_table)
    print(f"q_table saved to {output_path}")
    # record most recent q_table
    with open("./models/recent_model.id", 'w') as fw:
        fw.write(str(model_id))


def load_q_table(model_id=None):
    if model_id is None:
        id_file = "./models/recent_model.id"
        if not os.path.isfile(id_file):
            return None

        with open(id_file, 'r') as f:
            model_id = f.read()

    model_path = f"./models/q_table-{model_id}.npy"
    if os.path.isfile(model_path):
        print(f"load model from {model_path}")
        return np.load(model_path)
    else:
        return None


def main():
    # get environment
    env = get_env()

    # set episodes for rl
    EPISODES = 15000

    # initialize q-table
    # actions: {0, 1, 2}
    # observation: 
    #   P - position [-1.2, 0.6]
    #   V - velocity [-0.07, 0.07] 
    #   P x V
    # split P and V into discrete spaces slice_num_1d x slice_num_1d

    hyper_params = HyperParams(
        learning_rate=0.1, 
        discount_factor=0.95
    )

    # observations
    position_range = (-1.2, 0.6)
    velocity_range = (-.07, .07)
    slice_num_1d = 20
    action_dim = 3
    observation_dims = (20, 20)

    os_mapper = map_observation_to_state(
        observation_ranges=(position_range, velocity_range), 
        observation_dims=observation_dims)

    # q_table
    q_table_shape = observation_dims+(action_dim,)
    q_table = load_q_table()

    if q_table is None or q_table.shape != q_table_shape:
        print("initialize q_table randomly")
        q_table = np.random.uniform(low=-2, high=0, size=q_table_shape)
    else:
        print("load q_table from history")

    position_list = []
    velocity_list = []

    for episode in range(EPISODES):
        step = 0
        observation = env.reset()
        done = False
        score = 0

        while not done:

            # show results
            if episode > 0 and episode % 1000 == 0:
                env.render()

            prev_position = observation[0]
            old_state = os_mapper(observation)
            action = action_from_observation(observation, q_table, os_mapper)

            observation, reward, done, info = env.step(action)
            new_state = os_mapper(observation)

            # reward = calculate_reward(
            #     prev_position=prev_position,
            #     position=observation[0])

            score += reward

            learn_from_reward(
                reward, q_table, 
                old_state, new_state, 
                action, hyper_params)

            step += 1

        if episode > 0 and episode % 100 == 0:
            save_q_table(q_table, episode)
            print(q_table.shape)
            plot1 = plt.figure(1)
            draw_matrix(
                np.apply_along_axis(np.argmax, 2, q_table), 
                fig=plot1, ax = plot1.gca())
            plt.pause(0.01)
            plot1.clear()

        if step < 200:
            print(f"{episode}: we made it at step: {step}")
        # print(f"{episode}: done step: {step}")
        # print(f"{episode}: position: {observation[0]}")
        position_list.append(observation[0])
        velocity_list.append(observation[1])

        # plt.scatter(episode, observation[0], color='green') # position
        # plt.scatter(episode, observation[1]*10, color='red')  # velocity
        plot2 = plt.figure(2)
        plt.scatter(episode, score, color='blue') # score
        if episode % 1000 == 0:
            plt.pause(0.001)

    env.close()
    position_list = norm_array(position_list)
    velocity_list = norm_array(velocity_list, shift=False)
    plt.show()


if __name__ == "__main__":
    main()
