import numpy as np
import torch
from torch import nn


# TimeLSTM takes two inputs
# [batch, t, x]
# [batch, t]
# x = [x1, x2 ... xm] is the features we measure for x
# t is the dimension of time
class TimeLSTMLayer(nn.Module):
    
    # default
    DEFAULT_DEVICE = 'cpu';
    DEFAULT_LAYER_NUM = 6;
    
    # properties
    device = None;          # the device
    layer_num = None;       # the number of layers
    
    def __init__(self, *, device=DEFAULT_DEVICE, layer_num=DEFAULT_LAYER_NUM):
        pass