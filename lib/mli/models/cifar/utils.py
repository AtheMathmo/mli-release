import torch.nn as nn

from mli.models.layers import LIConv2d, LIBatchNorm2d
from mli.models import BaseModel


def conv3x3(in_planes, out_planes, stride=1):
    return LIConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)


class Downsampler(BaseModel, nn.Module):
    def __init__(self, inplanes, stride, use_batchnorm=True):
        super(Downsampler, self).__init__()
        self._use_batchnorm = use_batchnorm
        self.pool = nn.AvgPool2d(1, stride=stride)
        if use_batchnorm:
            self.bn0 = LIBatchNorm2d(inplanes)

    @property
    def use_batchnorm(self):
        return self._use_batchnorm
    
    def reset_bn(self):
        if self._use_batchnorm:
            self.bn0.reset_running_stats()

    def forward(self, x):
        x = self.pool(x)
        if self._use_batchnorm:
            x = self.bn0(x)
        return x
    
    def interpolated_forward(self, x, alpha, state1, state2, name):
        x = self.pool(x)
        if self._use_batchnorm:
            w_name = '{}.bn0.weight'.format(name)
            b_name = '{}.bn0.bias'.format(name)
            w1 = state1.get(w_name)
            w2 = state2.get(w_name)
            b1 = state1.get(b_name)
            b2 = state2.get(b_name)
            x = self.bn0.interpolated_forward(x, alpha, w1, w2, b1, b2)
        return x
