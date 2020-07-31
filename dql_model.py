import collections
import random

import numpy as np
import torch
from torch import nn

import config


class QNet(nn.Module):
    def __init__(self, num_inputs, num_classes):
        super(QNet, self).__init__()
    
        self.pipe = nn.Sequential(  
            nn.Linear(num_inputs, 10), 
            nn.Tanh(), 
            nn.Linear(10, 20), 
            nn.Tanh(), 
            nn.Linear(20, 20), 
            nn.Tanh(), 
            nn.Linear(20, 30), 
            nn.Tanh(), 
            nn.Linear(30, num_classes)
        )
        self.N = torch.diag(torch.tensor(
            [   1/(0.6+1.2), 
                1/(0.07*2)]))

        self.B = torch.tensor(
            [-1.2, -0.07])

    def forward(self, x):
        # normalization
        with torch.no_grad():
            x = torch.mm((x - self.B), self.N)
        #print(f"normalized x: {x}")
        return self.pipe(x)


class ModelTranner:
    def __init__(self, q_net, lr):
        self.q_net = q_net
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.q_net.parameters(), lr=lr)

    def compute_loss(self, xx, yy, q_hat):
        S = torch.tensor(
            np.concatenate(
                [xxx[0] for xxx in xx]).reshape(-1, 2), dtype=torch.float).to(config.DEVICE)
        A = torch.tensor([xxx[1] for xxx in xx], dtype=torch.int64).to(config.DEVICE)

        prob_y = self.q_net(S)
        pred_y = prob_y.gather(
            1, A.unsqueeze(-1)
        ).squeeze(-1)

        true_y = self.compute_target(xx, yy, q_hat)

        #print(f"prob_y: {prob_y}")
        #print(f"pred_y: {pred_y}")
        #print(f"true_y: {true_y}")
        return self.loss_func(pred_y, true_y)

    def compute_target(self, xx, yy, q_hat):
        """
        y = r if not done
        y = R + gamma * max(Q_hat(S))
                            m
        """
        with torch.no_grad():
            q_gamma = 0.95
            Sp = torch.tensor(np.concatenate([xxx[3] for xxx in xx]).reshape(-1, 2), 
                dtype=torch.float).to(config.DEVICE)
            Sp_v = q_hat(Sp).max(dim=1)
            masked_Sp_v = torch.Tensor(yy).to(config.DEVICE)* Sp_v.values
            masked_Sp_v = masked_Sp_v.detach()
            R = torch.Tensor([xxx[2] for xxx in xx]).to(config.DEVICE)
            Y = R + q_gamma * masked_Sp_v
        return Y

    def train(self, x_batch, y_batch, q_hat):
        # print(f"size of x in train: {len(x_batch)}")
        self.optimizer.zero_grad()
        loss = self.compute_loss(x_batch, y_batch, q_hat)
        loss.backward()
        # print(f"loss: {loss}")

        self.optimizer.step()

        return loss, [params.grad for params in self.q_net.parameters()]


class DataLoader:
    def __init__(self, x=None, y=None, batch_size=32, memory_size=20000):
        self.memory_size = memory_size

        if x is None and y is None:
            self.data_x = collections.deque(maxlen=self.memory_size)
            self.data_y = collections.deque(maxlen=self.memory_size)
        # else:
        #     self.data_x = x
        #     self.data_y = y

        self.batch_size = batch_size

    def append(self, x, y):
        self.data_x.append(x)
        #print(f"appending x[0][1]: {x[0][1]}")
        self.data_y.append(y)

    def clear(self):
        self.data_x = collections.deque(maxlen=self.memory_size)
        self.data_y = collections.deque(maxlen=self.memory_size)

    def next_batch(self):
        idxs = random.sample(range(len(self.data_x)), self.batch_size)
        x_batch = [self.data_x[ii] for ii in idxs]
        y_batch = [self.data_y[ii] for ii in idxs]
        return (x_batch, y_batch)
