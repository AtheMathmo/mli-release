import torch
import torch.nn as nn
import numpy as np

from mli.models import BaseModel
from mli.models.layers import LILinear
from .utils import conv3x3, Downsampler


__all__ = ["FixupResNet", "fixup_resnet20", "fixup_resnet32", "fixup_resnet44", "fixup_resnet56", 
           "fixup_resnet110"]


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample
    
    @property
    def use_batchnorm(self):
        return False
    
    def reset_bn(self):
        pass

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)
        return out
    
    def interpolate_pstate(self, alpha, state1, state2, layer_name, pname):
        pname = "{}.{}".format(layer_name, pname)
        return (1 - alpha) * state1[pname] + alpha * state2[pname]
    
    def interpolated_forward(self, x, alpha, state1, state2, name):
        identity = x

        bias1a = self.interpolate_pstate(alpha, state1, state2, name, "bias1a")
        bias1b = self.interpolate_pstate(alpha, state1, state2, name, "bias1b")
        conv1_wname = "{}.conv1.weight".format(name)
        conv1_bias = "{}.conv1.bias".format(name)

        out = self.conv1.interpolated_forward(
            x + bias1a, alpha,
            state1[conv1_wname], state2[conv1_wname],
            state1.get(conv1_bias), state2.get(conv1_bias),
        )
        out = self.relu(out + bias1b)

        bias2a = self.interpolate_pstate(alpha, state1, state2, name, "bias2a")
        bias2b = self.interpolate_pstate(alpha, state1, state2, name, "bias2b")
        scale = self.interpolate_pstate(alpha, state1, state2, name, "scale")

        conv2_wname = "{}.conv2.weight".format(name)
        conv2_bias = "{}.conv2.bias".format(name)
        out = self.conv2.interpolated_forward(
            out + bias2a, alpha,
            state1[conv2_wname], state2[conv2_wname],
            state1.get(conv2_bias), state2.get(conv2_bias),
        ) * scale + bias2b

        if self.downsample is not None:
            identity = self.downsample.interpolated_forward(x + bias1a, alpha, state1, state2, "{}.downsample".format(name))
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)
        return out


class FixupResNet(BaseModel, nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(FixupResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = LILinear(64, num_classes)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, LILinear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
    
    @property
    def use_batchnorm(self):
        return False
    
    def reset_bn(self):
        pass

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = Downsampler(self.inplanes, stride, False)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x + self.bias1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x + self.bias2)

        return x

    def interpolate_pstate(self, alpha, state1, state2, pname):
        return (1 - alpha) * state1[pname] + alpha * state2[pname]


    def interpolated_forward(self, x, alpha, state1, state2):
        c1_wname = "conv1.weight"
        c1_bname = "conv1.bias"
        x = self.conv1.interpolated_forward(
            x, alpha,
            state1[c1_wname], state2[c1_wname],
            state1.get(c1_bname), state2.get(c1_bname)
        )
        bias1 = self.interpolate_pstate(alpha, state1, state2, "bias1")
        x = self.relu(x + bias1)

        for i,l in enumerate(self.layer1):
            x = l.interpolated_forward(x, alpha, state1, state2, "layer1.{}".format(i))
        for i,l in enumerate(self.layer2):
            x = l.interpolated_forward(x, alpha, state1, state2, "layer2.{}".format(i))
        for i,l in enumerate(self.layer3):
            x = l.interpolated_forward(x, alpha, state1, state2, "layer3.{}".format(i))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f1_wname = "fc.weight"
        f1_bname = "fc.bias"
        bias2 = self.interpolate_pstate(alpha, state1, state2, "bias2")
        x = self.fc.interpolated_forward(
            x + bias2, alpha,
            state1[f1_wname], state2[f1_wname],
            state1.get(f1_bname), state2.get(f1_bname)
        )

        return x


def fixup_resnet20(**kwargs):
    """Constructs a Fixup-ResNet-20 model.

    """
    model = FixupResNet(FixupBasicBlock, [3, 3, 3], **kwargs)
    return model


def fixup_resnet32(**kwargs):
    """Constructs a Fixup-ResNet-32 model.

    """
    model = FixupResNet(FixupBasicBlock, [5, 5, 5], **kwargs)
    return model


def fixup_resnet44(**kwargs):
    """Constructs a Fixup-ResNet-44 model.

    """
    model = FixupResNet(FixupBasicBlock, [7, 7, 7], **kwargs)
    return model


def fixup_resnet56(**kwargs):
    """Constructs a Fixup-ResNet-56 model.

    """
    model = FixupResNet(FixupBasicBlock, [9, 9, 9], **kwargs)
    return model


def fixup_resnet110(**kwargs):
    """Constructs a Fixup-ResNet-110 model.

    """
    model = FixupResNet(FixupBasicBlock, [18, 18, 18], **kwargs)
    return model
