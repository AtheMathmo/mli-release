import unittest

import torch


class TestModels(unittest.TestCase):
    def test_act_fn(self):
        from mli.models import get_activation_function
        act_fns = [
            "relu",
            "sigmoid",
            "tanh",
            "identity"
        ]
        for act_fn in act_fns:
            get_activation_function(act_fn)

    def test_fcnet(self):
        from mli.models import FCNet
        net = FCNet(10, [1])
        x = torch.ones(5, 10)
        net.reset_bn()
        net.forward(x)
        state1 = net.state_dict()
        state2 = net.state_dict()
        net.interpolated_forward(x, 0.5, state1, state2)

    def test_lae(self):
        from mli.models import LinearAE
        net = LinearAE(10, 1)
        x = torch.ones(5, 10)
        net.reset_bn()
        net.forward(x)
        state1 = net.state_dict()
        state2 = net.state_dict()
        net.interpolated_forward(x, 0.5, state1, state2)

    def test_efficient_net(self):
        from mli.models import EfficientNet
        net = EfficientNet.from_name("efficientnet-b0")
        x = torch.ones(1, 3, 224, 224)
        net.reset_bn()
        net.forward(x)
        state1 = net.state_dict()
        state2 = net.state_dict()
        net.interpolated_forward(x, 0.5, state1, state2)

    def test_resnet18(self):
        from mli.models import resnet18
        net = resnet18()
        x = torch.ones(1, 3, 224, 224)
        net.reset_bn()
        net.forward(x)
        state1 = net.state_dict()
        state2 = net.state_dict()
        net.interpolated_forward(x, 0.5, state1, state2)

    def test_resnet20(self):
        from mli.models import resnet20
        net = resnet20()
        x = torch.ones(1, 3, 32, 32)
        net.reset_bn()
        net.forward(x)
        state1 = net.state_dict()
        state2 = net.state_dict()
        net.interpolated_forward(x, 0.5, state1, state2)

    def test_fixup_resnet20(self):
        from mli.models import fixup_resnet20
        net = fixup_resnet20()
        x = torch.ones(1, 3, 32, 32)
        net.reset_bn()
        net.forward(x)
        state1 = net.state_dict()
        state2 = net.state_dict()
        net.interpolated_forward(x, 0.5, state1, state2)

    def test_transformer(self):
        from mli.models import LITransformerModel
        net = LITransformerModel(100, 2, 2, 2, 2)
        x = torch.ones(32, 100).long()
        net.forward(x)
        state1 = net.state_dict()
        state2 = net.state_dict()
        net.interpolated_forward(x, 0.5, state1, state2)
