"""
To learn to write a simple linear model in pytorch
"""
import torch

# set weight and bias
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-1.0, requires_grad=True)

# define the forward function for the linear model
def forward(x):
    yhat = w*x + b
    return yhat

# define X
x = torch.tensor([[1.0],[2.0]])

# compute yhat
yhat = forward(x)
print("yhat",yhat)
