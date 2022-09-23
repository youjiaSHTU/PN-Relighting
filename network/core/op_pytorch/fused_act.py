## @author: limzh
## This file is the naive pytorch implementation for styleGAN3 op_cuda
## It avoids incomplilation due to cuda error.
import os
import sys
import numpy as np
import torch
import torch.nn as nn
# typing check
from torch import Tensor, tensor
import torch.nn.functional as F

# ########## Helper functions ###########
def fused_leaky_relu(self, x:Tensor, bias:Tensor, negative_slope:float=.2, scale:float=2**.5):
    # Merge input with bias
    if bias is not None:
        x = x + bias.view((1, -1) + (1,)*len(x.shape)-2)
    return scale * F.leaky_relu(x, negative_slope)
    
# FusedLeakyReLU -- pytorch Module version
class FusedLeakyReLU(nn.Module):
    """ FusedLeakyReLU = scale * LeakyReLu(x + bias)
    """
    def __init__(self, channel:int, negative_slop:float=.2, scale:float=2**.5) -> None:
        super().__init__()
        
        self.bias = nn.Parameter(
            torch.zeros(channel)
        )
        self.negative_slope = negative_slop
        self.scale = scale
    
    def forward(self, x:Tensor) -> Tensor:
        return fused_leaky_relu(x, self.bias, self.negative_slope, self.scale)