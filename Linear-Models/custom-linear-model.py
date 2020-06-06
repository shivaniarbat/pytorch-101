"""
A Custom Linear model using nn.Module
"""
from torch import nn
import torch

torch.manual_seed(1)

class LR(nn.Module):
    # constructor
    def __init__(self,input_size, output_size):
        # Inherit from parent
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size,output_size)

    # prediction function
    def forward(self, x):
        out = self.linear(x)
        return out

lr = LR(2,1)
print('the parameters:', list(lr.parameters()))