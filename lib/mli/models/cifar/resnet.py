import torch
import torch.nn as nn
import numpy as np

from mli.models import BaseModel
from mli.models.layers import LIBatchNorm2d, LILinear
from .utils import conv3x3, Downsampler


__all__ = ["ResNet", "resnet20", "resnet32", "resnet44", "resnet56", "resnet110"]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_batchnorm=True):
        super(BasicBlock, self).__init__()
        self._use_batchnorm = use_batchnorm
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if use_batchnorm:
            self.bn1 = LIBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if use_batchnorm:
            self.bn2 = LIBatchNorm2d(planes)
        else:
            self.res_scale = nn.Parameter(torch.zeros(1))
        self.downsample = downsample
    
    @property
    def use_batchnorm(self):
        return self._use_batchnorm
    
    def reset_bn(self):
        if self._use_batchnorm:
            self.bn1.reset_running_stats()
            self.bn2.reset_running_stats()
            if self.downsample:
                self.downsample.reset_bn()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.use_batchnorm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_batchnorm:
            out = self.bn2(out)
        else:
            out = self.res_scale * out

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out
    
    def interpolated_forward(self, x, alpha, state1, state2, name):
        identity = x

        conv1_wname = "{}.conv1.weight".format(name)
        conv1_bias = "{}.conv1.bias".format(name)

        out = self.conv1.interpolated_forward(
            x, alpha,
            state1[conv1_wname], state2[conv1_wname],
            state1.get(conv1_bias), state2.get(conv1_bias),
        )
        if self.use_batchnorm:
            wname = "{}.bn1.weight".format(name)
            bname = "{}.bn1.bias".format(name)

            out = self.bn1.interpolated_forward(
                out, alpha,
                state1.get(wname), state2.get(wname),
                state1.get(bname), state2.get(bname)
            )
        out = self.relu(out)

        conv2_wname = "{}.conv2.weight".format(name)
        conv2_bias = "{}.conv2.bias".format(name)
        out = self.conv2.interpolated_forward(
            out, alpha,
            state1[conv2_wname], state2[conv2_wname],
            state1.get(conv2_bias), state2.get(conv2_bias),
        )
        if self.use_batchnorm:
            wname = "{}.bn2.weight".format(name)
            bname = "{}.bn2.bias".format(name)

            out = self.bn2.interpolated_forward(
                out, alpha,
                state1.get(wname), state2.get(wname),
                state1.get(bname), state2.get(bname)
            )
        else:
            sname = "{}.res_scale".format(name)
            scale = (1-alpha) * state1[sname] + alpha * state2[sname]
            out = scale * out

        if self.downsample is not None:
            identity = self.downsample.interpolated_forward(x, alpha, state1, state2, "{}.downsample".format(name))
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


class ResNet(BaseModel, nn.Module):

    def __init__(self, block, layers, num_classes=10, use_batchnorm=True, identity_init=True):
        super(ResNet, self).__init__()
        self._use_batchnorm = use_batchnorm
        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        if use_batchnorm:
            self.bn1 = LIBatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LILinear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, LIBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if identity_init:
            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            for m in self.modules():
                if use_batchnorm and isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.res_scale, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = Downsampler(self.inplanes, stride, self.use_batchnorm)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_batchnorm=self.use_batchnorm))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes, use_batchnorm=self.use_batchnorm))

        return nn.Sequential(*layers)
    
    @property
    def use_batchnorm(self):
        return self._use_batchnorm
    
    def reset_bn(self):
        if self.use_batchnorm:
            self.bn1.reset_running_stats()
            for l in self.layer1:
                l.reset_bn()
            for l in self.layer2:
                l.reset_bn()
            for l in self.layer3:
                l.reset_bn()

    def forward(self, x):
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def interpolated_forward(self, x, alpha, state1, state2):
        c1_wname = "conv1.weight"
        c1_bname = "conv1.bias"
        x = self.conv1.interpolated_forward(
            x, alpha,
            state1[c1_wname], state2[c1_wname],
            state1.get(c1_bname), state2.get(c1_bname)
        )
        if self.use_batchnorm:
            w_name = "bn1.weight"
            b_name = "bn1.bias"
            w1 = state1.get(w_name)
            w2 = state2.get(w_name)
            b1 = state1.get(b_name)
            b2 = state2.get(b_name)
            x = self.bn1.interpolated_forward(x, alpha, w1, w2, b1, b2)
        x = self.relu(x)

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
        x = self.fc.interpolated_forward(
            x, alpha,
            state1[f1_wname], state2[f1_wname],
            state1.get(f1_bname), state2.get(f1_bname)
        )

        return x


def resnet20(**kwargs):
    """Constructs a ResNet-20 model.

    """
    model = ResNet(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32(**kwargs):
    """Constructs a ResNet-32 model.

    """
    model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44(**kwargs):
    """Constructs a ResNet-44 model.

    """
    model = ResNet(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56(**kwargs):
    """Constructs a ResNet-56 model.

    """
    model = ResNet(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110(**kwargs):
    """Constructs a ResNet-110 model.

    """
    model = ResNet(BasicBlock, [18, 18, 18], **kwargs)
    return model
