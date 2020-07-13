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


def main():
    # list all models
    # list_of_models = list_all_models()
    # print(list_of_models)
    mat = np.random.randn(10, 20)
    # draw_matrix(mat)
    # plt.show()


if __name__ == "__main__":
    main()