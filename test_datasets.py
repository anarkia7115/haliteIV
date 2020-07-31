import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, RandomSampler

from datasets import *


def test_circle_dataset():
    ds = BinaryCircleDataset()
    # ds = CircleDataset()
    ds.scatter()
    plt.show()


def test_replay_dataset():
    """
    Tests:
    1. random
    2. dynamic dataset
    3. maxlen has effect
    """
    ds = ReplayBuffer(maxlen=5)

    for i in range(10):
        i = i+100
        ds.append(
                Replay(state_t=i, best_action_t=0, 
                reward=0, state_tp1=0, done=0))

    # for row in ds:
    #     print(row)

    # random sampler
    dl = DataLoader(ds, sampler=RandomSampler(ds), batch_size=3)
    # print(dl)
    # check is random
    in_order = True
    prev_i = -1
    for batch in dl:
        for i in batch.state_t:
            if not prev_i < i.item():
                in_order = False
            prev_i = i

    assert not in_order

    # append some
    # get some, check new is loaded
    for i in range(10):
        i = i+200
        ds.append(
                Replay(state_t=i, best_action_t=0, 
                reward=0, state_tp1=0, done=0))
    for i in dl:
        m = max(i.state_t)

    assert m >= 200

    # append some
    # check maxlen takes effect
    for i in range(5):
        i = i+300
        ds.append(
                Replay(state_t=i, best_action_t=0, 
                reward=0, state_tp1=0, done=0))
        for i in dl:
            m = min(i.state_t)

    assert m >= 300


if __name__ == "__main__":
    # test_circle_dataset()
    test_replay_dataset()
