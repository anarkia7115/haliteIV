import glob

import torch
import numpy as np
import matplotlib.pyplot as plt


def list_all_models():
    model_dir = "./models"
    model_files = glob.glob(f"{model_dir}/*.npy")
    return model_files


def draw_matrix(data, fig, ax):
    psm = ax.pcolormesh(data, rasterized=True)
    fig.colorbar(psm, ax=ax)


def put_text_on_matrix(data, fig, ax):
    xdim = data.shape[0]
    ydim = data.shape[1]
    for xi in range(xdim):
        for yi in range(ydim):
            data[xi][yi]


def majority_action(action_list):
    b = np.bincount(action_list)
    return np.argmax(b)


def actions_along_features(q_table, action_axis, feature_names=None):
    """
    Result: 
        position 0, 1, 2, 3, 4
        action   0, 0, 1, 1, 2
        action - majority action taken

    Input: 
        q_table: (dim1, dim2, dim3, action_dim)
    """
    action_table = np.apply_along_axis(np.argmax, 
        axis=action_axis, arr=q_table)
    feature_num = action_table.ndim

    # initialize feature names if is None
    if feature_names is None:
        feature_names = list(range(feature_num))

    # apply_function, flatten + majority
    def _flatten_majority(arr):
        if arr.ndim > 1:
            arr_1d = np.ndarray.flatten(arr)
        else:
            arr_1d = arr
        return majority_action(arr_1d)

    for feature_axis in range(feature_num):
        actions = np.apply_along_axis(
            _flatten_majority, axis=feature_axis, arr=action_table)

        yield (feature_names[feature_axis], actions)


def plot_colorbar(ax, arr, label):
    ax.pcolormesh(np.reshape(arr, (-1, 1)).T, rasterized=True)
    ax.set_ylabel(label, fontsize=15)


def plot_actions_along_features(axs, q_table, action_axis_pos=None, feature_names=None):

    if action_axis_pos is None:
        action_axis_pos = q_table.ndim-1
    for ax, (feature_name, actions) in zip(axs, 
            actions_along_features(q_table, action_axis_pos, feature_names)):
        plot_colorbar(ax, actions, feature_name)


def cartesian_product(*array):
    la = len(array)
    target_arr_shape = [len(a) for a in array] + [la]
    arr = torch.empty(target_arr_shape)
    for i, a in enumerate(array):
        new_shape = [1] * la
        new_shape[i] = len(a)
        arr[...,i] = a.reshape(new_shape)

    return arr

def q_net_to_q_table(q_net, oss, steps=20):
    position_lin = torch.linspace(oss.low[0], oss.high[0], steps)
    velocity_lin = torch.linspace(oss.low[1], oss.high[1], steps)
    simulated_input = cartesian_product(position_lin, velocity_lin)
    return q_net(simulated_input)


def main():
    # list all models
    # list_of_models = list_all_models()
    # print(list_of_models)
    # mat = np.random.randn(10, 20)
    # draw_matrix(mat)
    # plt.show()
    pass


if __name__ == "__main__":
    # main()
    xx = torch.linspace(0, 1, 20)
    yy = torch.linspace(2, 3, 12)
    arr = cartesian_product(xx, yy)
    print(arr.shape)