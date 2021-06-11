"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.

   This code is based on
   [Luke Melas-Kyriazi's implementation](https://github.com/lukemelas/EfficientNet-PyTorch/tree/65671dda18c9158480d63978d833aae5dd705671/efficientnet_pytorch).
"""

## Adapted from:
# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import torch
from torch import nn
from torch.nn import functional as F
from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)

from mli.models import BaseModel
from mli.models.layers import LIConv2d, LIBatchNorm2d, LILinear


VALID_MODELS = (
    "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3",
    "efficientnet-b4", "efficientnet-b5", "efficientnet-b6", "efficientnet-b7",
    "efficientnet-b8",

    # Support the construction of "efficientnet-l2" without pretrained weights
    "efficientnet-l2"
)


class MBConvBlock(BaseModel, nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """
    def __init__(self, block_args, global_params, image_size=None, use_batchnorm=True):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self._use_batchnorm = use_batchnorm
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = LIBatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn"t modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = LIBatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = LIBatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    @property
    def use_batchnorm(self):
        return self._use_batchnorm

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock"s forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """
        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            if self.use_batchnorm:
                x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        if self.use_batchnorm:
            x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        if self.use_batchnorm:
            x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
    
    def reset_bn(self):
        if self._block_args.expand_ratio != 1:
            self._bn0.reset_running_stats()
        self._bn1.reset_running_stats()
        self._bn2.reset_running_stats()
    
    def interpolated_forward(self, inputs, alpha, state1, state2, idx, drop_connect_rate=None):
        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv.interpolated_forward(
                inputs, alpha,
                state1["_blocks.{}._expand_conv.weight".format(idx)],
                state2["_blocks.{}._expand_conv.weight".format(idx)],
                state1.get("_blocks.{}._expand_conv.bias".format(idx)),
                state2.get("_blocks.{}._expand_conv.bias".format(idx))
            )
            if self.use_batchnorm:
                w_name = "_blocks.{}._bn0.weight".format(idx)
                b_name = "_blocks.{}._bn0.bias".format(idx)
                w1 = state1.get(w_name)
                w2 = state2.get(w_name)
                b1 = state1.get(b_name)
                b2 = state2.get(b_name)
                x = self._bn0.interpolated_forward(x, alpha, w1, w2, b1, b2)
            x = self._swish(x)

        x = self._depthwise_conv.interpolated_forward(
            x, alpha,
            state1["_blocks.{}._depthwise_conv.weight".format(idx)],
            state2["_blocks.{}._depthwise_conv.weight".format(idx)],
            state1.get("_blocks.{}._depthwise_conv.bias".format(idx)),
            state2.get("_blocks.{}._depthwise_conv.bias".format(idx))
        )
        if self.use_batchnorm:
            w_name = "_blocks.{}._bn1.weight".format(idx)
            b_name = "_blocks.{}._bn1.bias".format(idx)
            w1 = state1.get(w_name)
            w2 = state2.get(w_name)
            b1 = state1.get(b_name)
            b2 = state2.get(b_name)
            x = self._bn1.interpolated_forward(x, alpha, w1, w2, b1, b2)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce.interpolated_forward(
                x_squeezed, alpha,
                state1["_blocks.{}._se_reduce.weight".format(idx)],
                state2["_blocks.{}._se_reduce.weight".format(idx)],
                state1.get("_blocks.{}._se_reduce.bias".format(idx)),
                state2.get("_blocks.{}._se_reduce.bias".format(idx))
            )
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand.interpolated_forward(
                x_squeezed, alpha,
                state1["_blocks.{}._se_expand.weight".format(idx)],
                state2["_blocks.{}._se_expand.weight".format(idx)],
                state1.get("_blocks.{}._se_expand.bias".format(idx)),
                state2.get("_blocks.{}._se_expand.bias".format(idx))
            )
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv.interpolated_forward(
            x, alpha,
            state1["_blocks.{}._project_conv.weight".format(idx)],
            state2["_blocks.{}._project_conv.weight".format(idx)],
            state1.get("_blocks.{}._project_conv.bias".format(idx)),
            state2.get("_blocks.{}._project_conv.bias".format(idx))
        )
        if self.use_batchnorm:
            w_name = "_blocks.{}._bn2.weight".format(idx)
            b_name = "_blocks.{}._bn2.bias".format(idx)
            w1 = state1.get(w_name)
            w2 = state2.get(w_name)
            b1 = state1.get(b_name)
            b2 = state2.get(b_name)
            x = self._bn2.interpolated_forward(x, alpha, w1, w2, b1, b2)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class EfficientNet(BaseModel, nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)
    """

    def __init__(self, blocks_args=None, global_params=None, use_batchnorm=True):
        super().__init__()
        assert isinstance(blocks_args, list), "blocks_args should be a list"
        assert len(blocks_args) > 0, "block args must be greater than 0"
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._use_batchnorm = use_batchnorm

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = LIBatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1: # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = LIBatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = LILinear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()
    
    @property
    def use_batchnorm(self):
        return self._use_batchnorm

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.

        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
        """
        endpoints = dict()

        # Stem
        x = self._conv_stem(inputs)
        if self.use_batchnorm:
            x = self._bn0(x)
        x = self._swish(x)
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints["reduction_{}".format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        x = self._conv_head(x)
        if self.use_batchnorm:
            x = self._bn1(x)
        x = self._swish(x)
        endpoints["reduction_{}".format(len(endpoints)+1)] = x

        return endpoints

    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._conv_stem(inputs)
        if self.use_batchnorm:
            x = self._bn0(x)
        x = self._swish(x)

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._conv_head(x)
        if self.use_batchnorm:
            x = self._bn1(x)
        x = self._swish(x)
        return x

    def forward(self, inputs):
        """EfficientNet"s forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        x = self.extract_features(inputs)
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x
    
    def reset_bn(self):
        self._bn0.reset_running_stats()
        for block in self._blocks:
            block.reset_bn()
        self._bn1.reset_running_stats()
    
    def interpolated_extract_features(self, inputs, alpha, state1, state2):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._conv_stem.interpolated_forward(
            inputs, alpha,
            state1["_conv_stem.weight"], state2["_conv_stem.weight"],
            state1.get("_conv_stem.bias"), state2.get("_conv_stem.bias")
        )
        if self.use_batchnorm:
            w_name = "_bn0.weight"
            b_name = "_bn0.bias"
            w1 = state1.get(w_name)
            w2 = state2.get(w_name)
            b1 = state1.get(b_name)
            b2 = state2.get(b_name)
            x = self._bn0.interpolated_forward(x, alpha, w1, w2, b1, b2)
        x = self._swish(x)

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block.interpolated_forward(x, alpha, state1, state2, idx, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._conv_head.interpolated_forward(
            x, alpha,
            state1["_conv_head.weight"], state2["_conv_head.weight"],
            state1.get("_conv_head.bias"), state2.get("_conv_head.bias")
        )
        if self.use_batchnorm:
            w_name = "_bn1.weight"
            b_name = "_bn1.bias"
            w1 = state1.get(w_name)
            w2 = state2.get(w_name)
            b1 = state1.get(b_name)
            b2 = state2.get(b_name)
            x = self._bn1.interpolated_forward(x, alpha, w1, w2, b1, b2)
        x = self._swish(x)
        return x
    
    def interpolated_forward(self, x, alpha, state1, state2):
        # Convolution layers
        x = self.interpolated_extract_features(x, alpha, state1, state2)
        # x = self.extract_features(x)
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)

            w1 = state1["_fc.weight"]
            w2 = state2["_fc.weight"]
            b1 = state1.get("_fc.bias")
            b2 = state2.get("_fc.bias")
            x = self._fc.interpolated_forward(x, alpha, w1, w2, b1, b2)
        return x

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            in_channels (int): Input data"s channel number.
            override_params (other key word params):
                Params to override model"s global_params.
                Optional key:
                    "width_coefficient", "depth_coefficient",
                    "image_size", "dropout_rate",
                    "num_classes", "batch_norm_momentum",
                    "batch_norm_epsilon", "drop_connect_rate",
                    "depth_divisor", "min_depth"

        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):
        """create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool):
                Whether to load pretrained weights
                trained with advprop (valid when weights_path is None).
            in_channels (int): Input data"s channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model"s global_params.
                Optional key:
                    "width_coefficient", "depth_coefficient",
                    "image_size", "dropout_rate",
                    "batch_norm_momentum",
                    "batch_norm_epsilon", "drop_connect_rate",
                    "depth_divisor", "min_depth"

        Returns:
            A pretrained efficientnet model.
        """
        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path, load_fc=(num_classes == 1000), advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError("model_name should be one of: " + ", ".join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """Adjust model"s first convolution layer to in_channels, if in_channels not equals 3.

        Args:
            in_channels (int): Input data"s channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
