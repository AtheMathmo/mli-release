import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import numpy as np

from mli.metrics import compute_avg_gauss_length


class InterpolatingModel(nn.Module):
    def __init__(self):
        super(InterpolatingModel, self).__init__()

    def interpolated_forward(self, x, alpha, state1, state2):
        pass


class LinearModel(InterpolatingModel):
    def __init__(self):
        super(LinearModel, self).__init__()

    def interpolated_forward(self, x, alpha, state1, state2):
        state = alpha * state1 + (1 - alpha) * state2
        return F.linear(x, state)


class ArcModel(InterpolatingModel):
    """Maps alpha to [sin(theta + x), cos(theta + x)]
    """
    def __init__(self):
        super(ArcModel, self).__init__()

    def interpolated_forward(self, x, alpha, state1, state2):
        theta = alpha * state1 + (1 - alpha) * state2
        y = torch.sin(theta + x)
        z = torch.cos(theta + x)
        return torch.stack([y, z], -1)


class RandomData(IterableDataset):
    def __init__(self, size, dim):
        super(RandomData).__init__()
        self.size = size
        self.dim = dim
        self.X = torch.randn(size, dim)

    def __iter__(self):
        for i in range(self.size):
            yield self.X[i], 0


class TestGaussLength(unittest.TestCase):
    def test_arc(self):
        model = ArcModel()
        theta1 = 0
        theta2 = np.pi
        dataset = RandomData(100, 1)
        random_loader = DataLoader(dataset, 10)
        arc_gl = compute_avg_gauss_length(model, theta1, theta2,
                                          np.linspace(0, 1, 200), random_loader)
        self.assertAlmostEqual(arc_gl, theta2, places=4)
