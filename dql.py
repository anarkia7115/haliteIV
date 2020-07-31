import os
import time
from collections import Counter

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import gym

from visualize_model import \
    plot_actions_along_features, q_net_to_q_table, \
    visualize_q_net
from dql_model import QNet, DataLoader, ModelTranner
from simulation import EnvRunner
import config

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
        # draw_matrix(
        #     np.apply_along_axis(np.argmax, 2, q_table), 
        #     fig=plot1, ax=ax_q_table)
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


def run_with_q_net(env, q_net, writer:SummaryWriter):
    done = False
    state_t = env.reset()
    step = 0

    while not done:
        env.render()
        with torch.no_grad():
            state_t = torch.tensor(state_t).unsqueeze(0).float()
            action_v = q_net(state_t)
            action = action_v.argmax().item()

            #print(f"state:{state_t}\taction_v:{action_v}\taction:{action}")
            writer.add_scalar("action", 
                action)
            writer.add_scalars("action_v", {
                "action_v0": action_v[0][0], 
                "action_v1": action_v[0][1], 
                "action_v2": action_v[0][2], 
                })

        state_t, _, done, _ = env.step(action)
        step += 1


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
    q_net = QNet(2, 3).to(config.DEVICE)
    q_hat = QNet(2, 3).to(config.DEVICE)
    q_hat.load_state_dict(q_net.state_dict())

    trainer = ModelTranner(q_net, lr=0.0001)

    # initialize env
    env = get_env()
    env.reset()

    # initialize tb writer
    writer = SummaryWriter()

    # initialize ml
    mini_batch_size = 128
    dl = DataLoader(batch_size=mini_batch_size)

    runner = EnvRunner(
        env, history=dl, q_hat=q_hat, epsilon=1, 
        writer=writer)

    warmup_steps = 5

    # initialize visualization
    fig1 = plt.figure(1, figsize=(15, 10))
    gs = fig1.add_gridspec(4, 6)
    ax_q_table = fig1.add_subplot(gs[:2, :2])
    nn_axs = [None, None, None]
    nn_axs[0] = fig1.add_subplot(gs[0:2, 2:4])
    nn_axs[1] = fig1.add_subplot(gs[2:4, 2:4])
    nn_axs[2] = fig1.add_subplot(gs[0:2, 4:6])

    action_axs = [fig1.add_subplot(gs[pos, :2])
        for pos in range(2, 4)]

    def _visualizer():
        visualize_q_net(fig1, nn_axs, 
            action_axs, ax_q_table, 
            q_net, env)

    # warmup
    runner.run_n_episode(warmup_steps)

    # run and train
    q_hat_update_period = 8000

    # run_with_q_net(env, q_net)
    """
    1. warm-up
    2. with every certain steps
        a. sample from dl
        b. backward loss and step for q_net
    3. update q_hat
    4. repeat n episodes
    """

    step = 0
    max_episode = 3000
    render_iter = 0
    for episode in range(max_episode):
        print(f"episode: {episode}")
        done = False
        state_t = env.reset()
        episode_step = 0
        while not done:
            done, state_tp1 = runner.run_one_step(state_t)
            state_t = state_tp1
            # sample
            xx, yy = dl.next_batch()
            #for row in xx:
            #    print(row)
            #print(xx)
            # backward
            loss, grad = trainer.train(xx, yy, q_hat)
            #print(f"grad:{grad}")

            if step % q_hat_update_period == q_hat_update_period - 1:
                q_hat.load_state_dict(q_net.state_dict())
                print("reassign q hat")

            step += 1
            episode_step += 1
            # end of while
        print(f"finish_step: {episode_step}")
        runner.epsilon_update(episode)

        for name, param in q_net.named_parameters():
            if "bias" not in name:
                writer.add_histogram(name, param)


        writer.add_scalar("loss", loss)
        writer.flush()
        print(f"loss: {loss}")
        #print(f"grad: {grad}")
        if episode % 10 == 9:
            run_with_q_net(env, q_net, writer)
            render_iter += 1

    env.close()


if __name__ == "__main__":
    main_dql()
