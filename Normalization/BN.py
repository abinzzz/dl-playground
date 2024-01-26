from torch import nn
import torch

# With Learnable Parameters
m = nn.BatchNorm2d(3)
# Without Learnable Parameters
m = nn.BatchNorm2d(3, affine=False)
input = torch.randn(2, 3, 4, 5)
output = m(input)

