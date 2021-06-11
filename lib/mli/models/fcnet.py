import functools

import torch
import torch.nn as nn

from .base import BaseModel
from .init import init_weights
from .layers import LILinear, LIBatchNorm1d


class FCNet(BaseModel, nn.Module):
    def __init__(self, insize, hsizes, act_fn=nn.ReLU, sigmoid_out=False, init_type="default", batch_norm=False):
        super(FCNet, self).__init__()

        assert len(hsizes) > 0, "Must specify at least one hidden layer size"
        self.insize = insize
        self.hsizes = hsizes
        self.act_fn = act_fn
        self.sigmoid_out = sigmoid_out
        self.init_type = init_type
        self.batch_norm = batch_norm

        layers = list()
        layers.append(LILinear(insize, hsizes[0]))

        for i in range(1, len(hsizes)):
            if batch_norm:
                layers.append(LIBatchNorm1d(hsizes[i - 1]))
            layers.append(act_fn())
            layers.append(LILinear(hsizes[i - 1], hsizes[i]))
        self.layers = nn.Sequential(*layers)

        self.apply(functools.partial(init_weights, init_type))

    @property
    def use_batchnorm(self):
        return self.batch_norm

    def reset_bn(self):
        for l in self.layers:
            if isinstance(l, LIBatchNorm1d):
                l.reset_running_stats()

    def forward(self, x):
        o = self.layers(x.view(x.shape[0], -1))
        if self.sigmoid_out:
            return torch.sigmoid(o)
        else:
            return o

    def interpolated_forward(self, x, alpha, state1, state2):
        o = x.view(x.shape[0], -1)
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], LILinear) or isinstance(self.layers[i], LIBatchNorm1d):
                w_name = "layers.{}.weight".format(str(i))
                b_name = "layers.{}.bias".format(str(i))
                o = self.layers[i].interpolated_forward(
                    o, alpha,
                    state1[w_name], state2[w_name],
                    state1.get(b_name), state2.get(b_name)
                )

            else:
                o = self.layers[i](o)

        if self.sigmoid_out:
            return torch.sigmoid(o)
        else:
            return o
