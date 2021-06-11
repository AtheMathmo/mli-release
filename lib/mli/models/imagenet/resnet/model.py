import torch
import torch.nn as nn

from mli.models import BaseModel
from mli.models.layers import LILinear, LIConv2d, LIBatchNorm2d


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return LIConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return LIConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self, inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        use_batchnorm=True
    ):
        self._use_batchnorm = use_batchnorm
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = LIBatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if use_batchnorm:
            self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if use_batchnorm:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    
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
        if self._use_batchnorm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self._use_batchnorm:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    def interpolated_forward(self, x, alpha, state1, state2, layer_name):
        identity = x

        out = self.conv1.interpolated_forward(
            x, alpha,
            state1["{}.conv1.weight".format(layer_name)],
            state2["{}.conv1.weight".format(layer_name)],
            state1.get("{}.conv1.bias".format(layer_name)),
            state2.get("{}.conv1.bias".format(layer_name))
        )
        if self._use_batchnorm:
            w_name = "{}.bn1.weight".format(layer_name)
            b_name = "{}.bn1.bias".format(layer_name)
            w1 = state1.get(w_name)
            w2 = state2.get(w_name)
            b1 = state1.get(b_name)
            b2 = state2.get(b_name)
            out = self.bn1.interpolated_forward(out, alpha, w1, w2, b1, b2)
        out = self.relu(out)

        out = self.conv2.interpolated_forward(
            out, alpha,
            state1["{}.conv2.weight".format(layer_name)],
            state2["{}.conv2.weight".format(layer_name)],
            state1.get("{}.conv2.bias".format(layer_name)),
            state2.get("{}.conv2.bias".format(layer_name))
        )
        if self._use_batchnorm:
            w_name = "{}.bn2.weight".format(layer_name)
            b_name = "{}.bn2.bias".format(layer_name)
            w1 = state1.get(w_name)
            w2 = state2.get(w_name)
            b1 = state1.get(b_name)
            b2 = state2.get(b_name)
            out = self.bn2.interpolated_forward(out, alpha, w1, w2, b1, b2)

        if self.downsample is not None:
            identity = self.downsample.interpolated_forward(
                x, alpha, state1, state2, "{}.downsample".format(layer_name)
            )

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Blank implementation for now
    """
    def __init__(self):
        super(Bottleneck, self).__init__()
    
    def forward(self, x):
        return x


class Downsampler(BaseModel, nn.Module):
    def __init__(self, inplanes, planes, expansion, stride, norm_layer=None, use_batchnorm=True):
        super(Downsampler, self).__init__()
        self._use_batchnorm = use_batchnorm
        if norm_layer is None:
            norm_layer = LIBatchNorm2d
        self.conv = conv1x1(inplanes, planes * expansion, stride)
        if use_batchnorm:
            self.norm_layer = norm_layer(planes * expansion)

    @property
    def use_batchnorm(self):
        return self._use_batchnorm
    
    def reset_bn(self):
        if self._use_batchnorm:
            self.norm_layer.reset_running_stats()

    def forward(self, x):
        x = self.conv(x)
        if self._use_batchnorm:
            x = self.norm_layer(x)
        return x
    
    def interpolated_forward(self, x, alpha, state1, state2, name):
        x = self.conv.interpolated_forward(
            x, alpha,
            state1["{}.conv.weight".format(name)], state2["{}.conv.weight".format(name)],
            state1.get("{}.conv.bias".format(name)), state2.get("{}.conv.bias".format(name))
        )
        if self._use_batchnorm:
            w_name = "{}.norm_layer.weight".format(name)
            b_name = "{}.norm_layer.bias".format(name)
            w1 = state1.get(w_name)
            w2 = state2.get(w_name)
            b1 = state1.get(b_name)
            b2 = state2.get(b_name)
            x = self.norm_layer.interpolated_forward(x, alpha, w1, w2, b1, b2)
        return x


class ResNet(BaseModel, nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        use_batchnorm=True
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = LIBatchNorm2d
        self._norm_layer = norm_layer
        self._use_batchnorm = use_batchnorm

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = LIConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if self._use_batchnorm:
            self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LILinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, LIConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (LIBatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks,
                    stride=1, dilate=False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsampler = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsampler = Downsampler(self.inplanes, planes, block.expansion, stride, norm_layer, self.use_batchnorm)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsampler, self.groups,
                            self.base_width, previous_dilation, norm_layer, self.use_batchnorm))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, use_batchnorm=self.use_batchnorm))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
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
            for l in self.layer4:
                l.reset_bn()

    def forward(self, x):
        return self._forward_impl(x)

    def interpolated_forward(self, x, alpha, state1, state2):
        x = self.conv1.interpolated_forward(
            x, alpha,
            state1["conv1.weight"], state2["conv1.weight"],
            state1.get("conv1.bias"), state2.get("conv1.bias")
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
        x = self.maxpool(x)
        for i, l in enumerate(self.layer1):
            x = l.interpolated_forward(x, alpha, state1, state2, "layer1.{}".format(i))
        for i, l in enumerate(self.layer2):
            x = l.interpolated_forward(x, alpha, state1, state2, "layer2.{}".format(i))
        for i, l in enumerate(self.layer3):
            x = l.interpolated_forward(x, alpha, state1, state2, "layer3.{}".format(i))
        for i, l in enumerate(self.layer4):
            x = l.interpolated_forward(x, alpha, state1, state2, "layer4.{}".format(i))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc.interpolated_forward(
            x, alpha,
            state1["fc.weight"], state2["fc.weight"],
            state1.get("fc.bias"), state2.get("fc.bias")
        )

        return x
