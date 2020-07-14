import os
import time

import matplotlib.pyplot as plt
import numpy as np
import gym

from visualize_model import draw_matrix, \
    plot_actions_along_features

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


def train_while_simulation(
        episode, env, os_mapper, 
        epsilon, q_table, hyper_params):

    step = 0
    score = 0
    done = False
    observation = env.reset()

    while not done:

        # show results
        if episode > 0 and episode % 1000 == 0:
            env.render()

        old_state = os_mapper(observation)

        # choose action
        if np.random.uniform(0, 1) > epsilon:
            action = action_from_observation(observation, q_table, os_mapper)
        else:
            action = env.action_space.sample()

        observation, reward, done, _ = env.step(action)
        new_state = os_mapper(observation)

        # reward = calculate_reward(
        #     prev_position=prev_position,
        #     position=observation[0])

        if observation[0] > 0.5:  # pass flag
            # reward = 1_000_000 / (step - 90) ** 2
            reward = 0

        score += reward

        learn_from_reward(
            reward, q_table, 
            old_state, new_state, 
            action, hyper_params)

        step += 1

    return step


def decay_episodes(epsilon, epsilon_decay, epsilon_min):
    if epsilon > epsilon_min:
        epsilon = epsilon * epsilon_decay
        # print(f"epsilon decayed to {epsilon}")

    return epsilon


def split_every_k_steps(steps, step_range):
    """
    indice_chunks: 0, 1, 2, 3; 4, 5, 6, 7; 8, 9
    ([indices[0] for indices in indice_chunks], 
     [[values[idx] for idx in indices] for indices in indice_chunks])
    """
    def _chunk_indices():
        indice_buffer = []
        for i in range(len(steps)):
            indice_buffer.append(i)
            if len(indice_buffer) == step_range:
                yield indice_buffer.copy()
                indice_buffer.clear()

        if len(indice_buffer) > 0:
            yield indice_buffer.copy()

    indice_chunks = list(_chunk_indices())

    return (
        [indices[0] for indices in indice_chunks], 
        [[steps[idx] for idx in indices] for indices in indice_chunks])


def show_episode_infomation(episode, q_table, steps, epsilon):

    # show information every 100 steps
    if episode > 0 and episode % 100 == 0:
        # show epsilon
        print(f"epsilon: {epsilon}")

        # save q table
        save_q_table(q_table, episode)

        # initialize axes
        feature_num = q_table.ndim-1
        grid_rows = feature_num*2+1

        plot1 = plt.figure(1, figsize=(4, 8))
        gs = plot1.add_gridspec(grid_rows, feature_num)
        ax_q_table = plot1.add_subplot(gs[:feature_num, :])
        ax_stats = plot1.add_subplot(gs[feature_num, :])
        action_axs = [plot1.add_subplot(gs[pos, :]) 
            for pos in range(feature_num+1, grid_rows)]
        assert len(action_axs) == feature_num

        # draw q table
        draw_matrix(
            np.apply_along_axis(np.argmax, 2, q_table), 
            fig=plot1, ax=ax_q_table)
        plt.pause(0.01)

        # aggregate steps info
        if episode < 1000:
            step_range = 50
        else:
            step_range = 500
        episode_x, step_groups = split_every_k_steps(
            [-ss for ss in steps], step_range)
        # max
        max_steps = [np.max(ss) for ss in step_groups]
        ax_stats.plot(episode_x, max_steps, label='max')

        # average
        mean_steps = [np.mean(ss) for ss in step_groups]
        ax_stats.plot(episode_x, mean_steps, label='avg')

        # min
        min_steps = [np.min(ss) for ss in step_groups]
        ax_stats.plot(episode_x, min_steps, label='min')

        ax_stats.legend(loc=2)


        # draw actions_along_features
        plot_actions_along_features(action_axs, q_table, 
            q_table.ndim - 1, feature_names=["position", "velocity"])

        plt.pause(0.01)
        plot1.clear()

    if steps[-1] < 200:
        print(f"{episode}: we made it at step: {steps[-1]}")

    if episode % 1000 == 0:
        plt.pause(0.001)


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
        learning_rate=0.06, 
        discount_factor=0.95
    )

    # observations
    position_range = (-1.2, 0.6)
    velocity_range = (-.07, .07)

    action_dim = 3
    observation_dims = (40, 40)

    # epsilon decay
    epsilon = 1
    epsilon_restore = 0.7
    epsilon_min = 0.05
    epsilon_decay = 0.999
    epsilon_restore_period = 5000

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

    # decay state
    epsilon_restored = False

    # stats
    steps = []

    for episode in range(EPISODES):
        step = train_while_simulation( 
            episode, env, os_mapper, 
            epsilon, q_table, hyper_params)

        # stats
        steps.append(step)
        if not epsilon_restored:
            if min(steps) < 200:  # not randomly hit yet
                epsilon = epsilon_restore
                epsilon_restored = True

        # restore epsilon periodically
        if episode % epsilon_restore_period == 0 and episode > 0:
            epsilon = epsilon_restore

        # decay epsilon
        epsilon = decay_episodes(epsilon, epsilon_decay, epsilon_min)

        show_episode_infomation(episode, q_table, steps, epsilon)

    env.close()
    # position_list = norm_array(position_list)
    # velocity_list = norm_array(velocity_list, shift=False)
    plt.show()


if __name__ == "__main__":
    main()
