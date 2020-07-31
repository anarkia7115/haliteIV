import numpy as np
import gym
import torch

from dql_model import DataLoader
import config


class EnvRunner:
    def __init__(self, env, history:DataLoader, q_hat, epsilon, writer):
        self.env = env
        self.history = history
        self.q_hat = q_hat
        self.epsilon = epsilon
        self.init_epsilon()

        self.total_round = 0
        self.max_pos = -999
        self.writer = writer

    def run_one_step(self, state_t):
        # check epsilon
        if np.random.uniform(0, 1) > self.epsilon:
            with torch.no_grad():
                action_v = self.q_hat.forward(
                        torch.Tensor(np.expand_dims(state_t, 0)).to(config.DEVICE)
                    )
                action = action_v.argmax().item()

                self.writer.add_scalar("action", 
                    action)
                self.writer.add_scalars("action_v", {
                    "action_v0": action_v[0][0], 
                    "action_v1": action_v[0][1], 
                    "action_v2": action_v[0][2], 
                    })
        else:
            # random action
            action = self.env.action_space.sample()

        # take one step
        state_tp1, reward, done, _ = self.env.step(action)

        # change reward
        pos = state_tp1[0]
        if pos >= 0.5:  # pass flag
            reward = 0
            print("reach final")
        #elif pos > self.max_pos:
        #    print(f"exceed max pos! {self.max_pos} -> {pos}")
        #    reward = (pos - self.max_pos) * 10
        #    self.max_pos = pos

        # save state (s, a, r, s’) into {replay}
        if not done:
            self.history.append(
                (state_t, action, reward, state_tp1), 1)
        else:
            state_tp1 = state_t
            self.history.append(
                (state_t, action, reward, state_t), 0)

        return done, state_tp1

    def run_one_episode(self, episode):
        done = False
        state_t = self.env.reset()

        while not done:
            done, state_t = self.run_one_step(state_t)

        self.total_round += 1

    def run_n_episode(self, N):
        for episode in range(N):
            self.run_one_episode(episode)
            self.epsilon_update(episode)

    def init_epsilon(self):
        self.epsilon_restore_period = 1000
        self.epsilon_restore = 0.2
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

    def epsilon_update(self, episode):
        # restore epsilon periodically
        if episode % self.epsilon_restore_period == \
                self.epsilon_restore_period - 1:
            print(f"{self.epsilon} restoreed to {self.epsilon_restore}")
            self.epsilon = self.epsilon_restore

        # decay epsilon
        new_epsilon = self.decay_epsilon()
        #if new_epsilon and self.total_round % 30 == 29:
        print(f"epsilon decayed to {new_epsilon}")

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
            return self.epsilon
        else:
            return self.epsilon


