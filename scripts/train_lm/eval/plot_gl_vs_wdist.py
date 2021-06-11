import os
import itertools
import argparse
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.style.use('seaborn')

import mli.metrics as metrics_utils
import json

from scripts.train_lm.eval.utils import *
from mli_eval.plotting.interp import plot_interp, plot_all_lm_interp
from mli_eval.processing.experiments import get_monotonicity_metrics, summarize_lm_metrics, get_run_lm_stats

parser = argparse.ArgumentParser()
parser.add_argument("--expdir", default="../runs/mli_lm/")
parser.add_argument("--outdir", default=".")
parser.add_argument("--plot_loss_lb", type=float, default=None)
args = parser.parse_args()

colors = sns.color_palette()

def model_cmap(config):
  model_name = config['model']
  if 'lstm' in model_name:
    return 'g'
  else:
    return 'purple'


def optim_cmap(config):
  optim_name = config['optimizer']
  if 'sgd' in optim_name:
    return 'g'
  # use_id_init = config['identity_init'] if 'identity_init' in config else True
  # if use_id_init:
  #   if 'nobn' in model_name:
  #     return 'b'
  #   else:
  #     return 'r'
  else:
    return 'purple'


def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
      paths = []
      for marker in m:
        if isinstance(marker, mmarkers.MarkerStyle):
          marker_obj = marker
        else:
          marker_obj = mmarkers.MarkerStyle(marker)
        path = marker_obj.get_path().transformed(
                    marker_obj.get_transform())
        paths.append(path)
      sc.set_paths(paths)
    else:
      raise ValueError("Invalid markers of length {} for data of length {}".format(len(m), len(x)))
    return sc

def process_experiments(expdir, outdir):
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  all_configs, all_metrics = get_run_lm_stats(expdir)
  summary = summarize_lm_metrics(
    all_configs, all_metrics,
    metric_summaries=[(np.min, 'val.interpolation.loss')]
  )
  with open(os.path.join(outdir, 'summary.json'), 'w') as f:
    json.dump(summary, f)

  monotonicity_summary = get_monotonicity_metrics(all_configs, all_metrics)
  print("ho")

  gauss_lengths = []
  bump_height = []
  loss = []
  color = []
  markers = []

  for run in monotonicity_summary:
      marker = "o"
      # min_loss = run['min_train_loss']
      try:
          bump = run['max_bump']
          gauss_lengths.append(run['avg_gl'])
          markers.append(marker)
          bump_height.append(bump)
          color.append("orange" if bump > 0 else colors[0])
      except:
          pass

  fig, ax = plt.subplots(figsize=(6, 3))
  ax.set_xlabel(r"$\log$ (Gauss length)", fontsize=14)
  ax.set_ylabel(r"$\min$ $\Delta$", fontsize=14)
  # ax.set_title("Gauss Length", fontsize=16)
  ax.set_title("Non-monotonicity against normalized Gauss length", fontsize=16)

  mscatter(np.log(gauss_lengths), bump_height, ax=ax, alpha=0.8, color=color, m=markers)
  plt.tight_layout()
  plt.show()

  wdist = []
  bump_height = []
  loss = []
  color = []
  markers = []
  for run in monotonicity_summary:
      marker = "o"
      # min_loss = run['min_train_loss']
      try:
          bump = run['max_bump']
          wdist.append(run['normed_weight_dist'])
          markers.append(marker)
          bump_height.append(bump)
          color.append("orange" if bump > 0 else colors[0])
      except:
          pass

  fig, ax = plt.subplots(figsize=(6, 3))
  ax.set_xlabel(r"$\log \Vert \theta_T - \theta_0 \Vert_2$ - $\log \Vert \theta_0 \Vert_2$", fontsize=14)
  ax.set_ylabel(r"$\min$ $\Delta$", fontsize=14)
  ax.set_title("Non-monotonicity against normalized weight distance", fontsize=16)

  mscatter(np.log(wdist), bump_height, ax=ax, alpha=0.8, color=color, m=markers)
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  process_experiments(args.expdir, args.outdir)
