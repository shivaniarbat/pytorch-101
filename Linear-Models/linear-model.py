"""
A Simple Linear model using Linear module in pytorch
"""
import torch
from torch.nn import Linear
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# set random seed
torch.manual_seed(1)

lr = Linear(in_features=2, out_features=1, bias=True)
print("Parameter w and b:", list(lr.parameters()))

# A method state_dict() Returns a Python dictionary
# object corresponding to the layers of each parameter tensor

print("Layers of each parameter:", lr.state_dict())

# make prediction now for the linear model
x = torch.tensor([[1.0,2.0],[2.0,2.0]])
yhat = lr(x)
print("The prediction:", yhat)
print(x.data)
