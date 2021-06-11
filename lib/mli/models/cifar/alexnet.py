# From: https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/alexnet.py

import torch.nn as nn

from mli.models.layers import LILinear, LIConv2d
from mli.models import BaseModel


class AlexNet(BaseModel, nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self._use_batchnorm = False

        self.features = nn.Sequential(
            LIConv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            LIConv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            LIConv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            LIConv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            LIConv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = LILinear(256, num_classes)

    @property
    def use_batchnorm(self):
        return self._use_batchnorm

    def reset_bn(self):
        pass

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def interpolated_forward(self, x, alpha, state1, state2):
        for i, l in enumerate(self.features):
            if isinstance(l, LIConv2d):
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



def alexnet(**kwargs):
    model = AlexNet(**kwargs)
    return model
