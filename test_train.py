import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn

from dql_model import DataLoader


def target_func(k, b):
    def _f(x):
        return k * x + b
    return _f

def generate_data(f):
    x = np.linspace(0, 2, 100)
    y = f(x) + np.random.normal(0, 0.2, 100)
    return x, y

def plot_model(model, x):
    y = model(torch.Tensor(x).reshape(-1, 1)
        ).detach()
    plt.plot(x, y)

def scatter(x, y):
    plt.scatter(x, y)

def test_train():
    """
    1. load data, plot data
    2. init model, plot model
    3. fit model
    4. goto xxx 
    """

    f = target_func(2, 1)

    fig = plt.figure()

    # load data
    x, y = generate_data(f)
    scatter(x, y)
    dl = DataLoader(x=x, y=y, batch_size=19)

    # init model
    l = nn.Linear(1, 1, bias=True)
    plot_model(l, x)

    loss_func = nn.MSELoss()

    plt.waitforbuttonpress()

    # fit model
    optimizer = torch.optim.SGD(l.parameters(), lr=0.1)
    for i in range(30):
        # get batch
        x_batch, y_batch = dl.next_batch()

        # scatter current batch
        plt.scatter(x_batch, y_batch, c='r')

        optimizer.zero_grad()

        # compute loss
        loss = loss_func(
            l(torch.Tensor(x_batch).reshape(-1, 1)), 
            torch.Tensor(y_batch).reshape(-1, 1))
        # update params
        loss.backward()
        optimizer.step()
        # plot model
        plot_model(l, x)
        fig.suptitle(i)
        plt.pause(0.1)

    # plot final result
    fig.clear()
    scatter(x, y)
    plot_model(l, x)
    params = list(l.parameters())
    k = params[0].item()
    b = params[1].item()
    fig.suptitle(f"k:{k}, b:{b}")
    plt.show()


if __name__ == "__main__":
    test_train()