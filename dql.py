import os
import time

import torch
import matplotlib.pyplot as plt
import numpy as np
import gym

from visualize_model import draw_matrix, \
    plot_actions_along_features, q_net_to_q_table
from dql_model import QNet, DataLoader, ModelTranner
from simulation import EnvRunner

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


def run_with_q_net(env, q_net):
    done = False
    state_t = env.reset()

    while not done:
        env.render()
        action = q_net.forward(
                torch.Tensor(np.expand_dims(state_t, 0))
            ).argmax().item()

        observation, reward, done, info = env.step(action)


def visualize_q_net(axs, q_net, env):

    plot_actions_along_features(axs, 
        q_net_to_q_table(q_net, env.observation_space))
    plt.pause(0.1)
    for ax in axs:
        ax.cla()

def main_dql():
    """
    1-3
        observation
        take action
        get reward
    save state (s, a, r, s’) into {replay}
    after {initial_action} steps, 
    sample {mini_batch_size} from replay, 
    compute y = r if session ends, else y = r + gamma * max_{a’}Q_hat(s’, a’)
    compute Loss y = (Q(s, a) - y)**2
    Update Q
    if {step} % {q_update_interval}, Q_hat = Q
    """

    # model
    q_net = QNet(2, 3)
    q_hat = QNet(2, 3)
    q_hat.load_state_dict(q_net.state_dict())

    trainer = ModelTranner(q_net, lr=0.01)

    # initialize env
    env = get_env()
    observation = env.reset()

    # initialize ml
    mini_batch_size = 256
    dl = DataLoader(batch_size=mini_batch_size)

    runner = EnvRunner(
        env, history=dl, q_hat=q_hat, epsilon=1)

    warmup_steps = 5

    # warmup
    runner.run_n_episode(warmup_steps)

    # run and train
    play_rounds = 3
    q_hat_update_period = 2

    run_with_q_net(env, q_net)
    fig, axs = plt.subplots(2, 1)

    visualize_q_net(axs, q_net, env)

    for epoch in range(1000):
        print(f"epoch: {epoch}")
        runner.run_n_episode(play_rounds)
        xx, yy = dl.next_batch()
        trainer.train(xx, yy, q_hat)

        # update q_hat periodically
        if epoch % q_hat_update_period == q_hat_update_period -1:
            q_hat.load_state_dict(q_net.state_dict())
            dl.clear()

        if epoch % 100 == 99:
            run_with_q_net(env, q_net)

        if epoch % 5 == 4:
            visualize_q_net(axs, q_net, env)

    env.close()


if __name__ == "__main__":
    main_dql()
