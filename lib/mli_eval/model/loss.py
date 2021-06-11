import torch.nn as nn
import torch.nn.functional as F
import abc


class EvalLossFn(abc.ABC):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def forward(self, inputs, logits, targets):
        return {
          "n": inputs.shape[0]
        }


class EvalNLL(EvalLossFn, nn.Module):
    def __init__(self, reduction="sum"):
        super(EvalNLL, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, logits, targets):
        loss = F.nll_loss(F.log_softmax(logits, dim=1), targets, reduction=self.reduction)
        if self.reduction != "none":
            loss = loss.item()
        return {
            "loss": loss,
            "n": inputs.shape[0]
        }


class EvalClassifierLoss(EvalLossFn, nn.Module):
    def __init__(self, reduction="sum"):
        super(EvalClassifierLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, logits, targets):
        preds = logits.argmax(1)
        acc = (preds == targets).float()
        if self.reduction == "mean":
            acc = acc.mean().item()
        elif self.reduction == "sum":
            acc = acc.sum().item()
        loss = F.nll_loss(F.log_softmax(logits, dim=1), targets, reduction=self.reduction)
        if self.reduction != "none":
            loss = loss.item()
        return {
            "loss": loss,
            "acc" : acc,
            "n": inputs.shape[0]
        }


class EvalAutoencoderLoss(EvalLossFn, nn.Module):
    def __init__(self, reduction="sum"):
        super(EvalAutoencoderLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, logits, targets):
        loss = F.mse_loss(
          logits.view(inputs.shape[0], -1),
          inputs.view(inputs.shape[0], -1),
          reduction=self.reduction
        )
        if self.reduction != "none":
            loss = loss.item()
        return {
            "loss": loss,
            "n": inputs.shape[0]
        }
