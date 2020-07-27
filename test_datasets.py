import matplotlib.pyplot as plt
from datasets import *


def test_circle_dataset():
    ds = CircleDataset()
    ds.scatter()
    plt.show()


if __name__ == "__main__":
    test_circle_dataset()