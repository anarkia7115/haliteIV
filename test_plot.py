import matplotlib.pyplot as plt
import numpy as np


def test_plot_and_clear():
    fig, axs = plt.subplots(2, 1)
    for i in range(10):
        axs[0].plot(range(10), np.random.randn(10))                                               
        axs[1].plot(range(10), np.random.randn(10))                                               
        plt.pause(0.25)
        for ax in axs:
            ax.cla()


def test_colorbar():
    fig, ax = plt.subplots()
    for i in range(10):
        psm = ax.pcolormesh(np.random.randn(10, 10), 
            rasterized=True)
        fig.colorbar(psm, ax=ax)
        plt.pause(1)
        fig.clear()
        ax.clear()


if __name__ == "__main__":
    test_colorbar()