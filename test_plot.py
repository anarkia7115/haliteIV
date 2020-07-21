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


if __name__ == "__main__":
    test_plot_and_clear()