import os
import itertools
import argparse
from collections import Counter

import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib.lines import Line2D

import mli.metrics as metrics_utils
import json

from .utils import *

parser = argparse.ArgumentParser()
parser.add_argument("expdir10")
parser.add_argument("expdir100")
parser.add_argument("--c10thresh", default=1.0, type=float)
parser.add_argument("--c100thresh", default=2.0, type=float)
args = parser.parse_args()

def get_summary(all_configs, all_metrics):
  summary = []
  for i in range(len(all_configs)):
    metrics = all_metrics[i]
    try:
      alphas = metrics['train.interpolation.alpha']['values']
      losses = metrics['train.interpolation.loss']['values']
      _, heights = metrics_utils.eval_monotonic(alphas, losses)
      if len(heights) > 0:
        max_bump = np.max(heights)
      else:
        max_bump = 0
      
      stats = {
        "run": all_configs[i]["run_id"],
        "max_bump": max_bump,
        "config": all_configs[i],
        "min_train_loss": np.min(metrics['train.loss']['values']),
        "final_train_loss": metrics['train.loss']['values'][-1],
        "weight_dist": metrics['weight_dist'],
        "normed_weight_dist": metrics['normed_weight_dist']
      }
    except:
      continue

    summary.append(
      stats
    )
  return summary

MODEL_MAP = [
  'NBN-F',
  'NBN-I',
  'BN',
  'BN-I'
]


def model_map(config):
  model_name = config['model_name']
  if 'fixup' in model_name:
    return 0
  use_id_init = config['identity_init'] if 'identity_init' in config else True
  if use_id_init:  
    if 'nobn' in model_name:
      return 1
    else:
      return 3
  else:
    return 2


def make_table(summary):
  cifar_table = {}
  for s in summary:
    config = s['config']
    if config['dset_name'] == 'cifar100':
      if s["final_train_loss"] > args.c100thresh:
        continue
    else:
      if s["final_train_loss"] > args.c10thresh:
        continue
    delta = s['max_bump']

    model_id = model_map(config)

    if model_id not in cifar_table:
      cifar_table[model_id] = {
        'sgd': {
          'delta_sum': 0,
          'non_mono': 0,
          'total': 0
        },
        'adam': {
          'delta_sum': 0,
          'non_mono': 0,
          'total': 0
        }
      }
    if config['optim_name'] == 'sgd':
      cifar_table[model_id]['sgd']['total'] += 1
      if delta > 0:
        cifar_table[model_id]['sgd']['non_mono'] += 1
        cifar_table[model_id]['sgd']['delta_sum'] += delta
    else:
      cifar_table[model_id]['adam']['total'] += 1
      if delta > 0:
        cifar_table[model_id]['adam']['non_mono'] += 1
        cifar_table[model_id]['adam']['delta_sum'] += delta
  
  model_keys = [2,3,1,0]
  
  toprow = "& & " + " & ".join([MODEL_MAP[m] for m in model_keys]) + r"\\\hline"
  sgdrow1 = r"\multirow{2}{*}{\rotatebox[origin=c]{90}{SGD}} & \% Non-monotonic & " + " & ".join(
    [
      "{:.2f} ({})".format(
        cifar_table[m]['sgd']['non_mono'] / cifar_table[m]['sgd']['total'],
        cifar_table[m]['sgd']['total']
      ) if cifar_table[m]['sgd']['total'] != 0 else '-' for m in model_keys
    ]
  ) + r"\rule{0pt}{2.6ex}\rule[-1.2ex]{0pt}{0pt}" + r"\\"
  sgdrow2 = r" & $\min \Delta$ & " + " & ".join(
    [
      "{:.3f}".format(
        cifar_table[m]['sgd']['delta_sum'] / cifar_table[m]['sgd']['non_mono']
      ) if cifar_table[m]['sgd']['non_mono'] != 0 else '0' for m in model_keys
    ]
  ) + r"\rule[-1.2ex]{0pt}{0pt}" + r"\\ \hline"
  adamrow1 = r"\multirow{2}{*}{\rotatebox[origin=c]{90}{Adam}} & \% Non-monotonic & " + " & ".join(
    [
      "{:.2f} ({})".format(
        cifar_table[m]['adam']['non_mono'] / cifar_table[m]['adam']['total'],
        cifar_table[m]['adam']['total']
      ) if cifar_table[m]['adam']['total'] != 0 else '-' for m in model_keys
    ]
  ) + r"\rule{0pt}{2.6ex}\rule[-1.2ex]{0pt}{0pt}" + r"\\ "
  adamrow2 = r" & $\min \Delta$ & " + " & ".join(
    [
      "{:.3f}".format(
        cifar_table[m]['adam']['delta_sum'] / cifar_table[m]['adam']['non_mono']
      ) if cifar_table[m]['adam']['non_mono'] != 0 else '0' for m in model_keys
    ]
  ) + r"\rule[-1.2ex]{0pt}{0pt}" + r"\\ \hline"
  
  print(toprow)
  print(sgdrow1)
  print(sgdrow2)
  print(adamrow1)
  print(adamrow2)

def make_lr_table(summary):
  lr_table = {}
  models = []
  for s in summary:
    config = s['config']
    lr = config['lr']
    if config['dset_name'] == 'cifar100':
      if s["final_train_loss"] > args.c100thresh:
        continue
    else:
      if s["final_train_loss"] > args.c10thresh:
        continue
    delta = s['max_bump']

    if lr not in lr_table:
      lr_table[lr] = {
        'sgd': {
          'bn': {
            'delta_sum': 0,
            'non_mono': 0,
            'total': 0
          },
          'nobn': {
            'delta_sum': 0,
            'non_mono': 0,
            'total': 0
          }
        },
        'adam': {
          'bn': {
            'delta_sum': 0,
            'non_mono': 0,
            'total': 0
          },
          'nobn': {
            'delta_sum': 0,
            'non_mono': 0,
            'total': 0
          }
        },
      }
    model = model_map(config)
    models.append(model)
    bn_str = 'bn' if model >= 2 else 'nobn'
    lr_table[lr][config['optim_name']][bn_str]['total'] += 1
    if delta > 0:
      lr_table[lr][config['optim_name']][bn_str]['non_mono'] += 1
      lr_table[lr][config['optim_name']][bn_str]['delta_sum'] += delta

  lrs = list(lr_table.keys())
  lrs.sort()
  print(Counter(models))

  toprow = "Learning Rate & &  " + " & ".join([str(l) for l in lrs]) + r"\\ \hline"
  sgdrow1 = r"\multirow{2}{*}{\rotatebox[origin=c]{90}{SGD}} & BN & " + " & ".join(
    [
      "{:.2f} ({})".format(
        lr_table[l]['sgd']['bn']['non_mono'] / lr_table[l]['sgd']['bn']['total'],
        lr_table[l]['sgd']['bn']['total']
      ) if lr_table[l]['sgd']['bn']['total'] != 0 else '-' for l in lrs
    ]
  ) + r"\rule{0pt}{2.6ex}\rule[-1.2ex]{0pt}{0pt}" + r"\\"
  sgdrow2 = r" & No BN & " + " & ".join(
    [
      "{:.2f} ({})".format(
        lr_table[l]['sgd']['nobn']['non_mono'] / lr_table[l]['sgd']['nobn']['total'],
        lr_table[l]['sgd']['nobn']['total']
      ) if lr_table[l]['sgd']['nobn']['total'] != 0 else '-' for l in lrs
    ]
  ) + r"\rule[-1.2ex]{0pt}{0pt}" + r"\\ \hline"
  adamrow1 = r"\multirow{2}{*}{\rotatebox[origin=c]{90}{Adam}} & BN & " + " & ".join(
    [
      "{:.2f} ({})".format(
        lr_table[l]['adam']['bn']['non_mono'] / lr_table[l]['adam']['bn']['total'],
        lr_table[l]['adam']['bn']['total']
      ) if lr_table[l]['adam']['bn']['total'] != 0 else '-' for l in lrs
    ]
  ) + r"\rule{0pt}{2.6ex}\rule[-1.2ex]{0pt}{0pt}" + r"\\"
  adamrow2 = r" & No BN & " + " & ".join(
    [
      "{:.2f} ({})".format(
        lr_table[l]['adam']['nobn']['non_mono'] / lr_table[l]['adam']['nobn']['total'],
        lr_table[l]['adam']['nobn']['total']
      ) if lr_table[l]['adam']['nobn']['total'] != 0 else '-' for l in lrs
    ]
  ) + r"\rule[-1.2ex]{0pt}{0pt}" + r"\\ \hline"
  print(toprow)
  print(sgdrow1)
  print(sgdrow2)
  print(adamrow1)
  print(adamrow2)

  sgd_D_sum = np.sum([
    lr_table[lr]['sgd']['bn']['delta_sum'] + lr_table[lr]['sgd']['nobn']['delta_sum'] for lr in lrs
  ])
  sgd_total = np.sum(
    [
      lr_table[lr]['sgd']['bn']['non_mono'] + lr_table[lr]['sgd']['nobn']['non_mono'] for lr in lrs
    ]
  )
  print()
  print("SGD Avg min D: {}".format(sgd_D_sum / sgd_total))

  adam_D_sum = np.sum([
    lr_table[lr]['adam']['bn']['delta_sum'] + lr_table[lr]['adam']['nobn']['delta_sum'] for lr in lrs
  ])
  adam_total = np.sum(
    [
      lr_table[lr]['adam']['bn']['non_mono'] + lr_table[lr]['adam']['nobn']['non_mono'] for lr in lrs
    ]
  )
  print()
  print("Adam Avg min D: {}".format(adam_D_sum / adam_total))

  bnrow = r"BN & " + " & ".join(
    [
      "{:.2f} ({})".format(
        (lr_table[l]['sgd']['bn']['non_mono'] + lr_table[l]['adam']['bn']['non_mono'])  / (lr_table[l]['sgd']['bn']['total'] + lr_table[l]['adam']['bn']['total']),
        (lr_table[l]['sgd']['bn']['total'] + lr_table[l]['adam']['bn']['total'])
      ) if (lr_table[l]['sgd']['bn']['total'] + lr_table[l]['adam']['bn']['total']) != 0 else '-' for l in lrs
    ]
  ) + r"\rule[-1.2ex]{0pt}{0pt}" + r"\\ \hline"
  nobnrow = r"BN & " + " & ".join(
    [
      "{:.2f} ({})".format(
        (lr_table[l]['sgd']['nobn']['non_mono'] + lr_table[l]['adam']['nobn']['non_mono'])  / (lr_table[l]['sgd']['nobn']['total'] + lr_table[l]['adam']['nobn']['total']),
        (lr_table[l]['sgd']['nobn']['total'] + lr_table[l]['adam']['nobn']['total'])
      ) if (lr_table[l]['sgd']['nobn']['total'] + lr_table[l]['adam']['nobn']['total']) != 0 else '-' for l in lrs
    ]
  ) + r"\rule[-1.2ex]{0pt}{0pt}" + r"\\ \hline"
  print()
  print()

  print(bnrow)
  print(nobnrow)

def process_experiments(expdir10, expdir100):
  c10_all_configs, c10_all_metrics = get_run_stats(expdir10)
  c100_all_configs, c100_all_metrics = get_run_stats(expdir100)
  c10_summary = get_summary(c10_all_configs, c10_all_metrics)
  c100_summary = get_summary(c100_all_configs, c100_all_metrics)
  all_summary = c10_summary + c100_summary
  print("Init/BN table:")
  print()
  make_table(all_summary)

  print()
  print()
  print("Optim/BN LR table:")
  print()
  make_lr_table(all_summary)


if __name__ == '__main__':
  process_experiments(args.expdir10, args.expdir100)