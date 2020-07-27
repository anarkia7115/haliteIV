import torch
from torch import nn
from dql_model import QNet


def main():
    input = torch.randn(1000, 5, requires_grad=True)
    target = torch.randint(5, (1000, ))
    m = QNet(5, 5)
    # m = nn.LogSoftmax(dim=-1)
    loss = nn.NLLLoss()
    # optimizer = torch.optim.SGD([input], lr=0.01)
    optimizer = torch.optim.SGD(m.parameters(), lr=0.01)

    for i in range(1000):
        output = loss(m(input), target)
        if i % 100 == 99:
            # print(input)
            print(output)
            for params in m.parameters():
                print(params.grad.shape)
                # print(params.grad)
        output.backward()
        optimizer.step()

if __name__ == "__main__":
    main()