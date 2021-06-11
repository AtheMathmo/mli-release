import torch
import functools

import torch.nn as nn
import torch.nn.functional as F

from .init import init_weights
from .base import BaseModel


def _squared_frob(weight):
    return (weight * weight).sum()


class LinearAE(BaseModel, nn.Module):
    def __init__(self, insize, hsize, init_type="default"):
        super(LinearAE, self).__init__()
        self._use_batchnorm = False

        self.insize = insize
        self.hsize = hsize
        self.init_type = init_type

        self.w1 = nn.Linear(insize, hsize, bias=False)
        self.w2 = nn.Linear(hsize, insize, bias=False)
        self.apply(functools.partial(init_weights, init_type))

    @property
    def use_batchnorm(self):
        return False

    def forward(self, x):
        o = self.w2(self.w1(x))
        return F.mse_loss(x, o)

    def reset_bn(self):
        pass

    def interpolated_forward(self, x, alpha, state1, state2):
        o = x.view(x.shape[0], -1)
        w1 = (1 - alpha) * state1["w1.weight"] + alpha * state2["w1.weight"]
        w2 = (1 - alpha) * state1["w2.weight"] + alpha * state2["w2.weight"]
        o = o.matmul(w1.t())
        return F.mse_loss(x, o.matmul(w2.t()))

    def uni_reg(self, l):
        return l * (_squared_frob(self.w1.weight) + _squared_frob(self.w2.weight))

    def non_uni_reg(self, L):
        return _squared_frob(torch.mm(L, self.w1.weight)) + _squared_frob(torch.mm(self.w2.weight, L))
