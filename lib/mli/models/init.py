import torch
import torch.nn as nn

import numpy as np


def init_weights(init_type, module):
    if init_type == "default":
        # use PyTorch default initialization (kaiming)
        pass

    elif init_type == "xavier":
        if type(module) == nn.Linear or type(module) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias:
                torch.nn.init.xavier_uniform_(module.bias)

    elif init_type == 'kaiming':
      if isinstance(module, nn.Linear):
        # TODO: Specialize init for other nonlinearities
        nn.init.kaiming_uniform_(module.weight, nonlinearity=nn.ReLU)
        module.bias.data.fill_(0.)

    elif init_type == "sparse":
        sparsity = 0.99
        gain = 1.0
        if type(module) == nn.Linear:
            stddev = gain * np.sqrt(
                2.0 / ((1.0 - sparsity) * (module.weight.data.shape[0] + module.weight.data.shape[1])))
            torch.nn.init.sparse_(module.weight.data, 0.8, stddev)
        else:
            raise ValueError("Unknown module for sparse initialization: {}".format(type(module)))

    else:
        raise ValueError("Invalid initialization: {}".format(init_type))
