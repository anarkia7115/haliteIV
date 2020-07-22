import matplotlib.pyplot as plt

from sklearn.metrics import precision_score
import numpy as np
import torch
from torch import nn

from dql_model import DataLoader
from visualize_model import visualize_q_net


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

def test_train_linear():
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

def circle_data_generator():
    """
        0 - in circle
        1 - on circle
        2 - out circle
        radius = 0.8
    """
    width = 0.3
    centerx = centery = center = 2
    radius = center * 1.5
    xx = np.random.randn(1000, 2)*radius + center
    def _func_y(point):
        if (point[0] - centerx)**2 + (point[1] - centery) **2 < (radius-width)**2:
            return 0
        elif (point[0] - centerx)**2 + (point[1] - centery) **2 < (radius+width)**2:
            return 1
        else:
            return 2
    yy = np.apply_along_axis(_func_y, axis=1, arr=xx)
    return xx, yy


def create_data_mesh(xx):
    range_x = (
        xx[...,0].min(), 
        xx[...,0].max()
    )

    range_y = (
        xx[...,1].min(), 
        xx[...,1].max()
    )

    X,Y = np.mgrid[
        range_x[0]:range_x[1]:0.1, 
        range_y[0]:range_y[1]:0.1]

    return np.stack([X, Y], axis=-1)


def test_train_circle():
    """
    1. generate circle data
    2. visualize circle data
    3. train on data
    4. visualize NN
    """
    """
    x, y E [0, 1] x [0, 1]
    """

    # generate data
    xx, yy = circle_data_generator()

    # show circle data
    fig2 = plt.figure(2, figsize=(10, 10))
    plt.scatter(xx[...,0], xx[...,1], c=yy)
    plt.pause(0.1)

    from dql_model import QNet
    q_net = QNet(2, 3)

    dl = DataLoader(xx, yy, batch_size=256)

    optimizer = torch.optim.SGD(q_net.parameters(), lr=0.001)
    loss_func = nn.NLLLoss()

    print(q_net)

    for i in range(1000):
        x_batch, y_batch = dl.next_batch()
        # plt.scatter(np.array(x_batch)[...,0], np.array(x_batch)[...,1], c=y_batch)
        y_lsm = q_net(torch.Tensor(x_batch))
        # y_pred = y_prob.argmax(dim=1)

        loss = loss_func(y_lsm, torch.tensor(y_batch, dtype=torch.int64))
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(i)
            print(f"loss: {loss}")
            precision = precision_score(
                y_batch, 
                q_net(torch.Tensor(x_batch)).argmax(dim=-1), 
                average='micro')
            print(f"precision: {precision}")

if __name__ == "__main__":
    test_train_circle()