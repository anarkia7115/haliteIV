import matplotlib.pyplot as plt

from dql import get_env
from dql_model import QNet
from visualize_model import *

def test_q_net_to_q_table():
    q_net = QNet(2, 3)
    env = get_env()
    q_table = q_net_to_q_table(q_net, env.observation_space).detach()

    print(q_table.shape)

    fig, axs = plt.subplots(2, 1)

    plot_actions_along_features(axs, q_table, q_table.ndim-1, ["position", "velocity"])

    plt.show()


if __name__ == "__main__":
    test_q_net_to_q_table()

