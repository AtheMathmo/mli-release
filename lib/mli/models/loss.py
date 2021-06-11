import torch.nn.functional as F

from functools import partial


LOSS_FN_MAP = {
    "ce": F.cross_entropy,
    "recon": partial(F.mse_loss, reduction="sum"),
}


def get_loss_fn(loss_name):
    return LOSS_FN_MAP[loss_name]
