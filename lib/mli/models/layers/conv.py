import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _pair


class LIConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        super(LIConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups,
            bias, padding_mode
        )
    
    def interpolated_forward(self, x, alpha, w1, w2, b1, b2):
        w = (1 - alpha) * w1 + alpha * w2
        if b1 is not None:
            b = (1 - alpha) * b1 + alpha * b2
        else:
            b = None
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            w, b, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(x, w, b, self.stride,
                        self.padding, self.dilation, self.groups)
