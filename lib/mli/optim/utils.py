import torch
import functools

from .kfac import KFACOptimizer


def get_optimizer(optim_name, lr, beta, **kwargs):
    optim_map = {
      'sgd': functools.partial(torch.optim.SGD, lr=lr, momentum=beta, **kwargs),
      'adam': functools.partial(torch.optim.Adam, lr=lr, betas=(beta, 0.999), **kwargs),
      'kfac': functools.partial(KFACOptimizer, lr=lr, momentum=beta, **kwargs)
    }
    if optim_name not in optim_map:
        raise ValueError("Unsupport optimizer: {}. Only 'sgd' or 'adam' "
                         "supported currently.".format(optim_name))
    return optim_map[optim_name]
