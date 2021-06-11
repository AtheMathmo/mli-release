import os
import copy
import itertools
import argparse
from functools import partial

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib.lines import Line2D
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mli.metrics as metrics_utils
from mli_eval.processing.experiments import get_monotonicity_metrics
import json

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("expdir")
parser.add_argument("outdir")
parser.add_argument("--thresh", type=float, default=30)
parser.add_argument("--show", action='store_true')
args = parser.parse_args()

def plot_interp_comparison(
  m1, m2, outdir,
  figname="interp_compare.png",
  figtitle="Monotonic linear interpolations",
  subtitles=None):
  fig, axs = plt.subplots(figsize=(8,3), nrows=1, ncols=2, sharex=True, sharey=True)
  steps_t1 = m1['train.interpolation.alpha']['values']
  losses_t1 = m1['train.interpolation.loss']['values']
  steps_e1 = m1['val.interpolation.alpha']['values']
  losses_e1 = m1['val.interpolation.loss']['values']
  axs[0].plot(steps_t1, losses_t1, alpha=0.9, c='blue')
  axs[0].plot(steps_e1, losses_e1, alpha=0.9, c='red')
  
  steps_t2 = m2['train.interpolation.alpha']['values']
  losses_t2 = m2['train.interpolation.loss']['values']
  steps_e2 = m2['val.interpolation.alpha']['values']
  losses_e2 = m2['val.interpolation.loss']['values']

  axs[1].plot(steps_t2, losses_t2, alpha=0.9, c='blue')
  axs[1].plot(steps_e2, losses_e2, alpha=0.9, c='red')
  fpath = os.path.join(outdir, figname)
  
  lhandles = [
    Line2D([0], [0], color='b', label='Train'),
    Line2D([0], [0], color='r', label='Test')
  ]
  axs[1].legend(handles=lhandles, loc='lower left', fontsize=14)
  axs[0].set_xlim(0, 1)
  axs[0].set_ylim(ymin=0)
  axs[0].set_xlabel(r"$\alpha$", size=14)
  axs[1].set_xlabel(r"$\alpha$", size=14)
  axs[0].set_ylabel("Loss", size=14)

  if subtitles is not None:
    axs[0].set_title(subtitles[0], fontsize=14)
    axs[1].set_title(subtitles[1], fontsize=14)
  plt.suptitle(figtitle, fontsize=16)
  plt.tight_layout()
  plt.savefig(fpath)
  if args.show:
    plt.show()
  plt.clf()
  plt.close()


def get_comparison_runs(metrics_summary, loss=True):
  min_mono_idx = 0
  nonmonos = []
  if loss:
    min_mono_loss = 1000
    for i,m in enumerate(metrics_summary):
      if m["max_bump"] == 0:
        if not np.isnan(m["min_train_loss"]) and m["min_train_loss"] < min_mono_loss:
          min_mono_loss = m["min_train_loss"]
          min_mono_idx = i
      else:
        if not np.isnan(m["min_train_loss"]) and m["min_train_loss"] < args.thresh:
          nonmonos.append(i)
  else:
    max_mono_acc = 0
    for i,m in enumerate(metrics_summary):
      if m["max_bump"] == 0:
        if not np.isnan(m["min_train_loss"]) and m["max_test_acc"] > max_mono_acc:
          max_mono_acc = m["max_test_acc"]
          min_mono_idx = i
      else:
        if not np.isnan(m["min_train_loss"]) and m["max_test_acc"] > args.thresh:
          nonmonos.append(i)

  print("Bumps: {}".format([metrics_summary[i]["max_bump"] for i in nonmonos]))
  nonmono_idx = np.argmax([metrics_summary[i]["max_bump"] for i in nonmonos])
  return min_mono_idx, nonmonos[nonmono_idx]


def process_experiments(expdir, outdir):
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  all_configs, all_metrics = get_run_stats(expdir)
  summary = get_monotonicity_metrics(all_configs, all_metrics)
  i,j = get_comparison_runs(summary, False)
  plot_interp_comparison(
    all_metrics[i], all_metrics[j],
    outdir, figtitle="Linear interpolations with the test set (CIFAR-10)",
    subtitles=["Train MLI holds", "Train MLI fails"],
    figname="holdout_compare.pdf"
  )


if __name__ == '__main__':
  process_experiments(args.expdir, args.outdir)