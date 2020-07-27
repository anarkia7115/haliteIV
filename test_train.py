import matplotlib.pyplot as plt

from sklearn.metrics import precision_score
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

from dql_model import QNet
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


def create_data_mesh(ds: Dataset):
    range_x = (
        ds.xx[...,0].min().item(), 
        ds.xx[...,0].max().item()
    )

    range_y = (
        ds.xx[...,1].min().item(), 
        ds.xx[...,1].max().item()
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
    print("creating data mesh")
    mesh = create_data_mesh(ds)

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


def load_circle_data():
    ds = CircleDataset(sample_num=20)

    mesh = create_data_mesh(ds)
    train_size = int(len(ds) * 0.8)
    test_size = len(ds) - train_size

    ds_tr, ds_te = random_split(ds, [train_size, test_size])
    dl_tr = DataLoader(ds_tr, batch_size=64)
    dl_te = DataLoader(ds_te, batch_size=64)

    return dl_tr, dl_te


def test_train_circle():
    dl_tr, dl_te = load_circle_data()
    q_net = QNet(2, 3)
    optimizer = torch.optim.SGD(q_net.parameters(), lr=0.001)

    loss_func = nn.NLLLoss()

    max_epoches = 100
    for epoch in range(max_epoches):
        for xx, yy in dl_tr:
            y_lsm = q_net(xx)
            loss_val = loss_func(y_lsm, yy)
            loss_val.backward()
            optimizer.step()
        if epoch % 10 == 9:
            print(f"epoch: {epoch}")
            print(f"loss: {loss_val}")
            print(f"y_lsm: {y_lsm}")
            print(f"xx: {xx}")
            print(f"yy: {yy}")
    


def test_train_circle_bak():
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

    mesh = create_data_mesh(ds)
    train_size = int(len(ds) * 0.8)
    test_size = len(ds) - train_size

    ds_tr, ds_te = random_split(ds, [train_size, test_size])
    dl_tr = DataLoader(ds_tr, batch_size=32)
    dl_te = DataLoader(ds_te, batch_size=32)

    # show circle data
    fig2 = plt.figure(2, figsize=(10, 10))
    for xx, yy in dl_tr:
        plt.scatter(xx[...,0], xx[...,1], c=yy)
    plt.pause(0.1)

    q_net = QNet(5, 3)

    optimizer = torch.optim.SGD(q_net.parameters(), lr=0.001)
    loss_func = nn.NLLLoss()

    # init grids for nn visualization
    plot3, gs = create_gs(3, 10, 10)
    layer_ids = [1, 3, 5, 7]
    ax_mat = init_model_layers_axs(q_net, plot3, gs, layer_ids)

    print(q_net)

    writer = SummaryWriter()

    max_epoches = 10000

    loss_id = 0
    for epoch in range(max_epoches):
        print(f"epoch: {epoch}")
        for xx, yy in dl_tr:
            y_lsm = q_net(xx)
            loss = loss_func(y_lsm, yy)
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss, loss_id)
            loss_id += 1

        with torch.no_grad():
            precision_list = []
            for xx, yy in dl_te:
                precision = precision_score(
                    yy, 
                    q_net(xx).argmax(dim=-1), 
                    average='micro')

                precision_list.append(precision)
            precision = np.mean(precision_list)
            writer.add_scalar('test_precision', precision, epoch)

            precision_list = []
            for xx, yy in dl_tr:
                precision = precision_score(
                    yy, 
                    q_net(xx).argmax(dim=-1), 
                    average='micro')

                precision_list.append(precision)
            precision = np.mean(precision_list)
            writer.add_scalar('train_precision', precision, epoch)


        # if i % 100 == 0:
        #     print(i)
        #     print(f"loss: {loss}")
        #     print(f"precision: {precision}")

        #     draw_model_layers(q_net, mesh, ax_mat, layer_ids)
        #     plt.pause(0.1)

        #     for axs in ax_mat:
        #         for ax in axs:
        #             ax.cla()

if __name__ == "__main__":
    test_train_circle()
    # test_draw_layer()