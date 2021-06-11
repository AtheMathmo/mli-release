import torch.nn as nn
import torch.nn.functional as F


class LILayerNorm(nn.modules.normalization.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LILayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)

    def interpolated_forward(self, x, alpha, w1, w2, b1, b2):
        if w1 is not None:
            w = (1 - alpha) * w1 + alpha * w2
        else:
            w = None
        if b1 is not None:
            b = (1 - alpha) * b1 + alpha * b2
        else:
            b = None
        return F.layer_norm(
            x, self.normalized_shape, w, b, self.eps)
