# From: https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py

import torch.nn.functional as F
import torch.nn as nn

from mli.models.layers import LILinear, LIConv2d
from mli.models import BaseModel


class LeNet(BaseModel, nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self._use_batchnorm = False

        self.conv1 = LIConv2d(3, 6, 5)
        self.conv2 = LIConv2d(6, 16, 5)
        self.fc1 = LILinear(16 * 5 * 5, 120)
        self.fc2 = LILinear(120, 84)
        self.fc3 = LILinear(84, num_classes)

    @property
    def use_batchnorm(self):
        return False

    def reset_bn(self):
        pass

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def interpolated_forward(self, x, alpha, state1, state2):
        wname = "conv1.weight"
        bname = "conv1.bias"
        out = self.conv1.interpolated_forward(x, alpha,
                                              state1[wname], state2[wname],
                                              state1[bname], state2[bname])
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        wname = "conv2.weight"
        bname = "conv2.bias"
        out = self.conv2.interpolated_forward(out, alpha,
                                              state1[wname], state2[wname],
                                              state1[bname], state2[bname])
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)

        wname = "fc1.weight"
        bname = "fc1.bias"
        out = self.fc1.interpolated_forward(out, alpha,
                                            state1[wname], state2[wname],
                                            state1[bname], state2[bname])
        out = F.relu(out)

        wname = "fc2.weight"
        bname = "fc2.bias"
        out = self.fc2.interpolated_forward(out, alpha,
                                            state1[wname], state2[wname],
                                            state1[bname], state2[bname])
        out = F.relu(out)

        wname = "fc3.weight"
        bname = "fc3.bias"
        out = self.fc3.interpolated_forward(out, alpha,
                                            state1[wname], state2[wname],
                                            state1[bname], state2[bname])
        return out
