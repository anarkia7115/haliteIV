from torch import nn
import torch


# loss_func = nn.L1Loss()
loss_func = nn.MSELoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.zeros(3, 5)

#for i in range(10): 
#    loss = loss_func(input, target) 
#    loss.backward() 
#    print(loss) 
#    print(input) 
#    with torch.no_grad(): 
#        input -= 0.3*input.grad 
#        input.grad.zero_()                                                                 

optimizer = torch.optim.SGD([input], lr=0.5)

for i in range(10):

    optimizer.zero_grad()  # clear grad

    loss = loss_func(input, target)
    loss.backward()
    print(loss)
    print(input)

    optimizer.step()  # update params