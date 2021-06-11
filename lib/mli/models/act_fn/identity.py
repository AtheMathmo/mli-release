import torch.nn as nn


class IdentityFn(nn.Module):
    def forward(self, x):
        return x
