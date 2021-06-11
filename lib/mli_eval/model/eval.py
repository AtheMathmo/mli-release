import torch
import numpy as np

from .loss import EvalClassifierLoss
from .utils import SumMetricsContainer


def eval_model(model, loader, loss=None, cuda=True):
    training = model.training
    model.eval()
    if loss is None:
        loss = EvalClassifierLoss(reduction="sum")
    metrics = SumMetricsContainer()
    with torch.no_grad():
        for x,y in loader:
            if cuda:
                x,y = x.cuda(), y.cuda()
            logits = model(x)
            b_metrics = loss(x, logits, y)
            metrics.update(b_metrics)
    model.train(training)
    return metrics.into_avg()


def eval_model_per_example(model, loader, loss=None, cuda=True, max_batches=None):
  training = model.training
  model.eval()
  all_metrics = {}
  ex_logits = []
  if loss is None:
      loss = EvalClassifierLoss(reduction="none")
  with torch.no_grad():
      batches = 0
      for x,y in loader:
          batches += 1
          if max_batches is not None and batches > max_batches:
              break
          if cuda:
              x,y = x.cuda(), y.cuda()
          logits = model(x)
          metrics = loss(x, logits, y)
          for k in metrics:
              if k not in all_metrics:
                  all_metrics[k] = []
              all_metrics[k].extend(metrics[k].cpu().numpy())
          ex_logits.extend(logits.cpu().numpy())
  model.train(training)
  return all_metrics, np.array(ex_logits)
