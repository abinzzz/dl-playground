import math
import torch
import torch.nn.functional as F


def gelu(x):
    """ 
    "GELU激活函数的原始实现
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    """ 
    GELU激活函数的当前实现
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def swish(x):
    """
    Swish 激活函数的实现
    """
    return x * torch.sigmoid(x)

activations = {"gelu": gelu, "relu": F.relu, "swish": swish, "gelu_new": gelu_new}
