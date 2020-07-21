import random

import numpy as np
import torch
from torch import nn


class QNet(nn.Module):
    def __init__(self, num_inputs, num_classes, dropout_prob=0):
        super(QNet, self).__init__()
    
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 5), 
            nn.ReLU(), 
            nn.Linear(5, 20), 
            nn.ReLU(), 
            nn.Linear(20, num_classes), 
            nn.Dropout(p=dropout_prob), 
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.pipe(x)


class ModelTranner:
    def __init__(self, q_net, lr):
        self.q_net = q_net
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.q_net.parameters(), lr=lr)

    def compute_loss(self, xx, yy, q_hat):
        S = np.concatenate([xxx[0] for xxx in xx]).reshape(-1, 2)
        A = torch.tensor([xxx[1] for xxx in xx], dtype=torch.int64)

        prob_y = self.q_net(torch.Tensor(S))
        pred_y = prob_y.gather(
            1, A.unsqueeze(-1)
        ).squeeze(-1)

        true_y = self.compute_target(xx, yy, q_hat)

        return self.loss_func(pred_y, true_y)

    def compute_target(self, xx, yy, q_hat):
        """
        y = r if not done
        y = R + gamma * max(Q_hat(S))
                            m
        """
        q_gamma = 0.95
        Sp = np.concatenate([xxx[3] for xxx in xx]).reshape(-1, 2)
        m = q_hat(torch.Tensor(Sp)).max(dim=1)
        masked_m = torch.Tensor(yy)* m.values
        R = torch.Tensor([xxx[2] for xxx in xx])
        Y = R + q_gamma * masked_m
        return Y

    def train(self, x_batch, y_batch, q_hat):
        print(f"size of x in train: {len(x_batch)}")
        self.optimizer.zero_grad()
        loss = self.compute_loss(x_batch, y_batch, q_hat)
        loss.backward()
        print(f"loss: {loss}")

        self.optimizer.step()


class DataLoader:
    def __init__(self, x=None, y=None, batch_size=32):
        if x is None and y is None:
            self.data_x = []
            self.data_y = []
        else:
            self.data_x = x
            self.data_y = y

        self.batch_size = batch_size

    def append(self, x, y):
        self.data_x.append(x)
        self.data_y.append(y)

    def clear(self):
        self.data_x = []
        self.data_y = []

    def next_batch(self):
        idxs = random.sample(range(len(self.data_x)), self.batch_size)
        x_batch = [self.data_x[ii] for ii in idxs]
        y_batch = [self.data_y[ii] for ii in idxs]
        return (x_batch, y_batch)
