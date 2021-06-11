import os
import itertools
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from matplotlib.lines import Line2D

import mli.metrics as metrics_utils
from mli_eval.processing.experiments import get_monotonicity_metrics
import json

from .utils import *

parser = argparse.ArgumentParser()
parser.add_argument("expdir")
parser.add_argument("outdir")
parser.add_argument("--thresh", default=0.1, type=float)
args = parser.parse_args()


def make_table(summary):
  lr_table = {}
  for s in summary:
    config = s["config"]
    lr = config["lr"]
    if s["final_train_loss"] > args.thresh:
      continue
    delta = s["max_bump"]

    if lr not in lr_table:
      lr_table[lr] = {
        "sgd": {
          "bn": {
            "delta_sum": 0,
            "non_mono": 0,
            "total": 0
          },
          "nobn": {
            "delta_sum": 0,
            "non_mono": 0,
            "total": 0
          }
        },
        "adam": {
          "bn": {
            "delta_sum": 0,
            "non_mono": 0,
            "total": 0
          },
          "nobn": {
            "delta_sum": 0,
            "non_mono": 0,
            "total": 0
          }
        },
      }
    bn_str = "bn" if config["use_batchnorm"] else "nobn"
    lr_table[lr][config["optim_name"]][bn_str]["total"] += 1
    if delta > 1e-3:
      lr_table[lr][config["optim_name"]][bn_str]["non_mono"] += 1
      lr_table[lr][config["optim_name"]][bn_str]["delta_sum"] += delta

  lrs = list(lr_table.keys())
  lrs.sort()
  

  toprow = "Learning Rate & &  " + " & ".join([str(l) for l in lrs]) + r"\\ \hline"
  sgdrow1 = r"\multirow{2}{*}{\rotatebox[origin=c]{90}{SGD}} & BN & " + " & ".join(
    [
      "{:.2f} ({})".format(
        lr_table[l]["sgd"]["bn"]["non_mono"] / lr_table[l]["sgd"]["bn"]["total"],
        lr_table[l]["sgd"]["bn"]["total"]
      ) if lr_table[l]["sgd"]["bn"]["total"] != 0 else "-" for l in lrs
    ]
  ) + r"\rule{0pt}{2.6ex}\rule[-1.2ex]{0pt}{0pt}" + r"\\"
  sgdrow2 = r" & No BN & " + " & ".join(
    [
      "{:.2f} ({})".format(
        lr_table[l]["sgd"]["nobn"]["non_mono"] / lr_table[l]["sgd"]["nobn"]["total"],
        lr_table[l]["sgd"]["nobn"]["total"]
      ) if lr_table[l]["sgd"]["nobn"]["total"] != 0 else "-" for l in lrs
    ]
  ) + r"\rule[-1.2ex]{0pt}{0pt}" + r"\\ \hline"
  adamrow1 = r"\multirow{2}{*}{\rotatebox[origin=c]{90}{Adam}} & BN & " + " & ".join(
    [
      "{:.2f} ({})".format(
        lr_table[l]["adam"]["bn"]["non_mono"] / lr_table[l]["adam"]["bn"]["total"],
        lr_table[l]["adam"]["bn"]["total"]
      ) if lr_table[l]["adam"]["bn"]["total"] != 0 else "-" for l in lrs
    ]
  ) + r"\rule{0pt}{2.6ex}\rule[-1.2ex]{0pt}{0pt}" + r"\\"
  adamrow2 = r" & No BN & " + " & ".join(
    [
      "{:.2f} ({})".format(
        lr_table[l]["adam"]["nobn"]["non_mono"] / lr_table[l]["adam"]["nobn"]["total"],
        lr_table[l]["adam"]["nobn"]["total"]
      ) if lr_table[l]["adam"]["nobn"]["total"] != 0 else "-" for l in lrs
    ]
  ) + r"\rule[-1.2ex]{0pt}{0pt}" + r"\\ \hline"
  print(toprow)
  print(sgdrow1)
  print(sgdrow2)
  print(adamrow1)
  print(adamrow2)

  sgd_D_sum = np.sum([
    lr_table[lr]["sgd"]["bn"]["delta_sum"] + lr_table[lr]["sgd"]["nobn"]["delta_sum"] for lr in lrs
  ])
  sgd_total = np.sum(
    [
      lr_table[lr]["sgd"]["bn"]["non_mono"] + lr_table[lr]["sgd"]["nobn"]["non_mono"] for lr in lrs
    ]
  )
  print()
  print("SGD Avg min D: {}".format(sgd_D_sum / sgd_total))

  adam_D_sum = np.sum([
    lr_table[lr]["adam"]["bn"]["delta_sum"] + lr_table[lr]["adam"]["nobn"]["delta_sum"] for lr in lrs
  ])
  adam_total = np.sum(
    [
      lr_table[lr]["adam"]["bn"]["non_mono"] + lr_table[lr]["adam"]["nobn"]["non_mono"] for lr in lrs
    ]
  )
  print()
  print("Adam Avg min D: {}".format(adam_D_sum / adam_total))




def process_experiments(expdir, outdir):
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  all_configs, all_metrics = get_run_stats(expdir)
  summary = get_monotonicity_metrics(all_configs, all_metrics)
  make_table(summary)


if __name__ == "__main__":
  process_experiments(args.expdir, args.outdir)
