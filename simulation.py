import numpy as np
import gym
import torch

from dql_model import DataLoader
import config


class EnvRunner:
    def __init__(self, env, history:DataLoader, q_hat, epsilon):
        self.env = env
        self.history = history
        self.q_hat = q_hat
        self.epsilon = epsilon
        self.init_epsilon()

        self.total_round = 0

    def run_one_step(self, state_t):
        # check epsilon
        if np.random.uniform(0, 1) > self.epsilon:
            with torch.no_grad():
                action = self.q_hat.forward(
                        torch.Tensor(np.expand_dims(state_t, 0)).to(config.DEVICE)
                    ).argmax().item()
        else:
            action = self.env.action_space.sample()

        # take one step
        state_tp1, reward, done, _ = self.env.step(action)

        # change reward
        if state_tp1[0] > 0.5:  # pass flag
            reward = 0

        # save state (s, a, r, s’) into {replay}
        if not done:
            self.history.append(
                (state_t, action, reward, state_tp1), 1)
        else:
            self.history.append(
                (state_t, action, reward, state_t), 0)

        return done

    def run_one_episode(self, episode):
        done = False
        state_t = self.env.reset()

        while not done:
            done = self.run_one_step(state_t)

        self.total_round += 1

    def run_n_episode(self, N):
        for episode in range(N):
            self.run_one_episode(episode)
            self.epsilon_update(episode)

    def init_epsilon(self):
        self.epsilon_restore_period = 1000
        self.epsilon_restore = 0.2
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9

    def epsilon_update(self, episode):
        # restore epsilon periodically
        if episode % self.epsilon_restore_period == \
                self.epsilon_restore_period - 1:
            print(f"{self.epsilon} restoreed to {self.epsilon_restore}")
            self.epsilon = self.epsilon_restore

        # decay epsilon
        new_epsilon = self.decay_epsilon()
        if new_epsilon and self.total_round % 30 == 29:
            print(f"epsilon decayed to {new_epsilon}")

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
            return self.epsilon
        else:
            return None


