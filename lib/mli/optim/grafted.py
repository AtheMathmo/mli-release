import torch

from torch.optim.optimizer import Optimizer
from collections import defaultdict


class GraftedOptimizer(Optimizer):
    def __init__(self, stepsize_optimizer, direction_optimizer, min_step_size=1e-5):
        self.stepsize_optimizer = stepsize_optimizer
        self.direction_optimizer = direction_optimizer
        self.min_step_size = min_step_size

        self.state = defaultdict(dict)
        # Cache the current optimizer parameters
        for group in stepsize_optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["cached_params"] = torch.zeros_like(p.data)
                param_state["cached_params"].copy_(p.data)

    def __getstate__(self):
        return {
            "state": self.state,
            "stepsize_optimizer": self.stepsize_optimizer,
            "direction_optimizer": self.direction_optimizer,
        }

    def zero_grad(self):
        self.stepsize_optimizer.zero_grad()
        self.direction_optimizer.zero_grad()

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError

    @property
    def param_groups(self):
        return self.step_optimizer.param_groups

    def get_step_size(self, optimizer):
        norm = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]["cached_params"] # old state
                diff = p.data - param_state
                norm += torch.sum(diff ** 2)
        return (norm ** 0.5).item()

    def step(self, closure=None):
        loss = self.stepsize_optimizer.step(closure)
        step_size = self.get_step_size(self.stepsize_optimizer)

        # need to reset parameters
        for group in self.direction_optimizer.param_groups:
            for p in group["params"]:
                p.data.copy_(self.state[p]["cached_params"])
        direction_loss = self.direction_optimizer.step(closure)
        direction_step_size = self.get_step_size(self.direction_optimizer)
        corrected_step_size = step_size / max(direction_step_size, self.min_step_size)

        # calculate corrected step
        for group in self.direction_optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]["cached_params"] # old state
                diff = p.data - param_state
                self.state[p]["cached_params"] = self.state[p]["cached_params"] \
                                                 + corrected_step_size * diff
                p.data.copy_(self.state[p]["cached_params"])
        return loss
