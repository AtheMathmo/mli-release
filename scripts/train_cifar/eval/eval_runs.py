import os
import itertools
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.style.use('seaborn')

import mli.metrics as metrics_utils
import json

from .utils import *
from mli_eval.plotting.interp import plot_interp, plot_all_interp
from mli_eval.processing.experiments import get_monotonicity_metrics, summarize_metrics

parser = argparse.ArgumentParser()
parser.add_argument("expdir")
parser.add_argument("outdir")
parser.add_argument("--plot_loss_lb", type=float, default=None)
args = parser.parse_args()

def model_cmap(config):
  model_name = config['model_name']
  if 'fixup' in model_name:
    return 'g'
  use_id_init = config['identity_init'] if 'identity_init' in config else True
  if use_id_init:  
    if 'nobn' in model_name:
      return 'b'
    else:
      return 'r'
  else:
    return 'purple'


def process_experiments(expdir, outdir):
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  all_configs, all_metrics = get_run_stats(expdir)
  summary = summarize_metrics(
    all_configs, all_metrics,
    metric_summaries=[(np.min, 'train.loss'), (np.max, 'train.acc'), (np.min, 'eval.interpolation.loss')]
  )
  with open(os.path.join(outdir, 'summary.json'), 'w') as f:
    json.dump(summary, f)

  monotonicity_summary = get_monotonicity_metrics(all_configs, all_metrics)
  with open(os.path.join(outdir, 'monotonicity_metrics.json'), 'w') as f:
    json.dump(monotonicity_summary, f)
  plot_all_interp(
    all_configs, all_metrics, outdir,
    loss_lb=args.plot_loss_lb, train_prefix="train",
    figname="all_interp_train.pdf",
    figtitle="Linear interpolations of CIFAR10 networks (Train)",
    legend_handles=[
        Line2D([0], [0], color='g', label='Fixup'),
        Line2D([0], [0],color='r', label='BatchNorm'),
        Line2D([0], [0],color='b', label='No BatchNorm'),
    ],
    conf_filter=lambda c: c['dset_name'] == 'cifar100',
    color_map=model_cmap
  )
  plot_all_interp(
    all_configs, all_metrics, outdir,
    loss_lb=args.plot_loss_lb, train_prefix="eval",
    figname="all_interp_test.pdf",
    figtitle="Linear interpolations of CIFAR10 networks (Test)",
    legend_handles=[
        Line2D([0], [0], color='g', label='Fixup'),
        Line2D([0], [0],color='r', label='BatchNorm'),
        Line2D([0], [0], color='purple', label='No BatchNorm'),
        Line2D([0], [0], color='b', label='No BatchNorm (Zero Init)'),
    ],
    conf_filter=lambda c: c['dset_name'] == 'cifar100',
    color_map=model_cmap
  )

  plot_all_interp(
    all_configs, all_metrics, outdir,
    loss_lb=args.plot_loss_lb, train_prefix="train",
    figname="all_interp_train_sgd.png",
    figtitle="Linear interpolations of CIFAR10 networks (SGD)",
    legend_handles=[
        Line2D([0], [0], color='g', label='Fixup'),
        Line2D([0], [0],color='r', label='BatchNorm'),
        Line2D([0], [0], color='purple', label='No BatchNorm'),
        Line2D([0], [0], color='b', label='No BatchNorm (Zero Init)'),
    ],
    conf_filter=lambda c: c['optim_name'] == 'sgd',
    color_map=model_cmap
  )

  plot_all_interp(
    all_configs, all_metrics, outdir,
    loss_lb=args.plot_loss_lb, train_prefix="train",
    figname="all_interp_train_adam.png",
    figtitle="Linear interpolations of CIFAR10 networks (Adam)",
    legend_handles=[
        Line2D([0], [0], color='g', label='Fixup'),
        Line2D([0], [0],color='r', label='BatchNorm'),
        Line2D([0], [0], color='purple', label='No BatchNorm'),
        Line2D([0], [0], color='b', label='No BatchNorm (Zero Init)'),
    ],
    conf_filter=lambda c: c['optim_name'] == 'adam',
    color_map=model_cmap
  )
  if False:
    for i in range(len(all_configs)):
      plot_interp(all_configs[i], all_metrics[i], outdir)



if __name__ == '__main__':
  process_experiments(args.expdir, args.outdir)