import matplotlib.pyplot as plt

from sklearn.metrics import precision_score
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split

from dql_model import QNet
from dql_model import DataLoader
from visualize_model import visualize_q_net
from datasets import CircleDataset


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


def draw_layer(layer, mesh, axs):
    neuron_id = 0
    for ax in axs:
        ax.imshow(
            layer(torch.Tensor(mesh)
            )[...,neuron_id].detach().numpy(), 
            aspect="auto")
        neuron_id += 1


def test_draw_layer():
    print("initializing q_net")
    q_net = QNet(2, 3)
    print("generating data")
    ds = CircleDataset()
    xx = ds.xx
    print("creating data mesh")
    mesh = create_data_mesh(xx)

    print("initialize gridspec")
    plot0 = plt.figure(0, figsize=(30, 30))

    gs = plot0.add_gridspec(20, 20)
    gs.update(wspace=0, hspace=0) # set the spacing between axes. 

    print("start plotting")
    draw_model_layers(q_net, mesh, plot0, gs)

    plot0.tight_layout()

    plt.waitforbuttonpress()


def create_gs(plot_id, rows, cols):
    plot = plt.figure(plot_id, figsize=(20, 20))

    gs = plot.add_gridspec(rows, cols)
    gs.update(wspace=0, hspace=0) # set the spacing between axes. 

    return plot, gs


def draw_model_layers(model, mesh, ax_mat, layer_ids):

    for axs, layer_id in zip(ax_mat, layer_ids):
        layer = model.pipe[:layer_id]
        draw_layer(layer, mesh, axs)


def init_model_layers_axs(model, plot, gs, layer_ids):
    """
    1. create gridspec
        fig size:  layer_num * max(neuron_num)
    2. loop through pipe
    3. loop through neuron
    4. imshow neuron
    """

    ax_mat = []
    for layer_id in layer_ids:

        layer = model.pipe[:layer_id]

        axs = [
            plot.add_subplot(gs[layer_id, i])
            for i in range(layer[-1].out_features) ]

        # turn axis off
        for ax in axs:
            ax.axis('off')

        ax_mat.append(axs)

    return ax_mat


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
    ds = CircleDataset()

    mesh = create_data_mesh(ds.xx)
    train_size = int(len(xx) * 0.8)
    test_size = len(xx) - train_size

    ds = random_split(torch.stack([xx, yy], -1, [train_size, test_size]))
    # xx = ds[0].dataset[...,0]
    # xx = ds[0].dataset[...,0]

    # random split
    random_split()

    # show circle data
    fig2 = plt.figure(2, figsize=(10, 10))
    plt.scatter(xx[...,0], xx[...,1], c=yy)
    plt.pause(0.1)

    q_net = QNet(2, 3)

    dl = DataLoader(xx, yy, batch_size=256)

    optimizer = torch.optim.SGD(q_net.parameters(), lr=0.001)
    loss_func = nn.NLLLoss()

    # init grids for nn visualization
    plot3, gs = create_gs(3, 10, 10)

    layer_ids = [1, 3, 5, 7]
    ax_mat = init_model_layers_axs(q_net, plot3, gs, layer_ids)

    print(q_net)

    writer = SummaryWriter(log_dir="runs")

    for i in range(10000):
        x_batch, y_batch = dl.next_batch()
        # plt.scatter(np.array(x_batch)[...,0], np.array(x_batch)[...,1], c=y_batch)
        y_lsm = q_net(torch.Tensor(x_batch))
        # y_pred = y_prob.argmax(dim=1)

        loss = loss_func(y_lsm, torch.tensor(y_batch, dtype=torch.int64))
        loss.backward()
        optimizer.step()
        writer.scalar('loss', loss)

        precision = precision_score(
            y_batch, 
            q_net(torch.Tensor(x_batch)).argmax(dim=-1), 
            average='micro')

        if i % 100 == 0:
            print(i)
            print(f"loss: {loss}")
            print(f"precision: {precision}")

            draw_model_layers(q_net, mesh, ax_mat, layer_ids)
            plt.pause(0.1)

            for axs in ax_mat:
                for ax in axs:
                    ax.cla()

if __name__ == "__main__":
    test_train_circle()
    # test_draw_layer()