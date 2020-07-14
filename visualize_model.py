import glob

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


def action_along_value(q_table, action_axis, feature_names=None):
    """
    Result: 
        position 0, 1, 2, 3, 4
        action   0, 0, 1, 1, 2
        action - majority action taken

    Input: 
        q_table: (dim1, dim2, dim3, action_dim)
    """
    action_table = np.apply_along_axis(np.argmax, axis=action_axis, arr=q_table)
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
        action_along_feature = np.apply_along_axis(
            _flatten_majority, axis=feature_axis, arr=action_table)

    pass


def main():
    # list all models
    # list_of_models = list_all_models()
    # print(list_of_models)
    mat = np.random.randn(10, 20)
    # draw_matrix(mat)
    # plt.show()


if __name__ == "__main__":
    main()