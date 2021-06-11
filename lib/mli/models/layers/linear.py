import torch.nn as nn
import torch.nn.functional as F


class LILinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LILinear, self).__init__(in_features, out_features, bias)
    
    def interpolated_forward(self, x, alpha, w1, w2, b1, b2):
        w = (1 - alpha) * w1 + alpha * w2
        if b1 is not None:
            b = (1 - alpha) * b1 + alpha * b2
        else:
            b = None
        return F.linear(x, w, b)
