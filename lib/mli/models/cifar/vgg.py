# From: https://github.com/pomonam/param-interpolate/blob/master/models/cifar/vgg.py

import math

import torch.nn as nn

from mli.models.layers import LIBatchNorm2d, LILinear, LIConv2d
from mli.models import BaseModel


model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
}


class VGG(BaseModel, nn.Module):
    def __init__(self, features, num_classes=10, use_batchnorm=False, **kwargs):
        super(VGG, self).__init__()
        self._use_batchnorm = use_batchnorm
        self.features = features
        self.classifier = LILinear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @property
    def use_batchnorm(self):
        return self._use_batchnorm

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, LIConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, LIBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, LILinear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def reset_bn(self):
        for l in self.features:
            if isinstance(l, LIBatchNorm2d):
                l.reset_running_stats()

    def interpolated_forward(self, x, alpha, state1, state2):
        for i, l in enumerate(self.features):
            if isinstance(l, LIBatchNorm2d) or isinstance(l, LIConv2d):
                wname = "features.{}.weight".format(str(i))
                bname = "features.{}.bias".format(str(i))

                x = l.interpolated_forward(x, alpha,
                                           state1[wname], state2[wname],
                                           state1[bname], state2[bname])
            else:
                x = l(x)

        x = x.view(x.size(0), -1)
        wname = "classifier.weight"
        bname = "classifier.bias"
        x = self.classifier.interpolated_forward(x, alpha,
                                                 state1[wname], state2[wname],
                                                 state1[bname], state2[bname])
        return x


def make_layers(cfg, use_batchnorm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = LIConv2d(in_channels, v, kernel_size=3, padding=1)
            if use_batchnorm:
                layers += [conv2d, LIBatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def vgg16(**kwargs):
    model = VGG(make_layers(cfg["D"], use_batchnorm=False),
                use_batchnorm=False, **kwargs)
    return model


def vgg16_bn(**kwargs):
    model = VGG(make_layers(cfg["D"], use_batchnorm=True),
                use_batchnorm=True, **kwargs)
    return model


def vgg19(**kwargs):
    model = VGG(make_layers(cfg["E"], use_batchnorm=False),
                use_batchnorm=False, **kwargs)
    return model


def vgg19_bn(**kwargs):
    model = VGG(make_layers(cfg["E"], use_batchnorm=True),
                use_batchnorm=True, **kwargs)
    return model
