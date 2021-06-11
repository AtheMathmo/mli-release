import torch.nn as nn
from .identity import IdentityFn


ACT_FN_MAP = {
  "identity": IdentityFn,
  "relu": nn.ReLU,
  "sigmoid": nn.Sigmoid,
  "tanh": nn.Tanh
}


def get_activation_function(act_fn_name):
    return ACT_FN_MAP[act_fn_name]
