import os
import argparse

import json

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")

import torch


from mli.models import FCNet, interpolate_state, get_activation_function
from mli.data import load_data
from mli.metrics.gauss_len import compute_avg_gauss_length, compute_avg_gauss_length_bn
import mli.metrics as metrics_utils

from .utils import get_run_stats, interp_networks_eval_examples, load_model_and_data


parser = argparse.ArgumentParser()
parser.add_argument("expdir")
parser.add_argument("outdir")
parser.add_argument("-d", "--data_eval_size", type=int, default=None)
parser.add_argument("-a", "--alphas", type=int, default=50)
parser.add_argument("--compute_gl", action="store_true")
parser.add_argument("--linearity_int", action="store_true")
args = parser.parse_args()


def save_summary_stats(all_configs, all_metrics, outdir):
  max_acc = 0
  max_acc_idx = 0
  min_acc = 100
  min_acc_idx = 0
  min_loss = 10**5
  min_loss_idx = 0
  max_loss = 0
  max_loss_idx = 0
  for i in range(len(all_configs)):
    try:
      metrics = all_metrics[i]
      min_loss_i = np.min(metrics["train.loss"]["values"])
      max_acc_i = np.max(metrics["train.acc"]["values"])
    except:
      continue
    if max_acc_i > max_acc:
      max_acc = max_acc_i
      max_acc_idx = i
    if max_acc_i < min_acc:
      min_acc = max_acc_i
      min_acc_idx = i
    if min_loss_i < min_loss:
      min_loss = min_loss_i
      min_loss_idx = i
    if min_loss_i > max_loss:
      max_loss = min_loss_i
      max_loss_idx = i
  min_loss_params = {
    "lr": all_configs[min_loss_idx]["lr"],
    "hsizes": all_configs[min_loss_idx]["hsizes"],
    "loss": min_loss
  }
  max_loss_params = {
    "lr": all_configs[max_loss_idx]["lr"],
    "hsizes": all_configs[max_loss_idx]["hsizes"],
    "loss": max_loss
  }
  max_acc_params = {
    "lr": all_configs[max_acc_idx]["lr"],
    "hsizes": all_configs[max_acc_idx]["hsizes"],
    "acc": max_acc
  }
  min_acc_params = {
    "lr": all_configs[min_acc_idx]["lr"],
    "hsizes": all_configs[min_acc_idx]["hsizes"],
    "loss": min_acc
  }
  stats = {
    "min_loss": min_loss_params,
    "max_loss": max_loss_params,
    "max_acc": max_acc_params,
    "min_acc": min_acc_params
  }
  with open(os.path.join(outdir, "summary.json"), "w") as f:
    json.dump(stats, f)


def save_monotonicity_metrics(all_configs, all_metrics, outdir):
  summary = []
  for i in range(len(all_configs)):
    try:
      metrics = all_metrics[i]
      alphas = metrics["interpolation.alpha"]["values"]
      losses = metrics["interpolation.loss"]["values"]
    except:
      continue
    widths, heights = metrics_utils.eval_monotonic(alphas, losses)
    if len(heights) > 0:
      max_bump = np.max(heights)
    else:
      max_bump = 0

    stats = {
      "run": all_configs[i]["run_id"],
      "max_bump": max_bump,
      "config": all_configs[i],
      "min_train_loss": np.min(metrics["train.loss"]["values"]),
      "final_train_loss": metrics["train.loss"]["values"][-1],
      "weight_dist": metrics["weight_dist"],
      "normed_weight_dist": metrics["normed_weight_dist"]
    }
    if args.linearity_int:
      stats["linearity_integral"] = metrics["linearity_integral"]

    if args.compute_gl:
      stats["gauss_length"] = metrics["gauss_length"]
    summary.append(
      stats
    )
  with open(os.path.join(outdir, "monotonicity_metrics.json"), "w") as f:
    json.dump(summary, f)


def eval_logit_linearity(logits, alphas):
  init_logits = logits[0]
  final_logits = logits[-1]
  diff = final_logits - init_logits

  diff_normalized = diff / np.sqrt((diff * diff).sum(-1, keepdims=True))
  # a.b = |a| |b| cos t

  # need |a| sin t

  # |a| ^2 sin^2 t = |a| ^ 2 ( 1- cos^2 t)
  # cos^2 t
  dists = []

  for (i, a) in enumerate(alphas):
    logit_step = logits[i] - init_logits
    proj = (logit_step * diff_normalized).sum(-1, keepdims = True) * diff_normalized
    anti_proj = logit_step - proj
    dist = (anti_proj * anti_proj).sum(-1)
    dists.append(dist)
  dists = np.array(dists).mean(-1)

  integral = 0.0
  for i in range(alphas.shape[0] - 1):
    integral += (dists[i + 1] + dists[i]) * (alphas[i+1] - alphas[i]) / 2.0
  # plt.semilogy(alphas, dists)
  # plt.show()
  return integral


def process_experiments(expdir, outdir):
  try:
    os.makedirs(outdir)
  except:
    pass
  all_configs, all_metrics = get_run_stats(expdir)
  for i in range(len(all_configs)):
    config = all_configs[i]
    hsizes = config["hsizes"]
    dset_name = config["dset_name"]
    act_fn = get_activation_function(config["act_fn"])
    use_batchnorm = config["use_batchnorm"] if "use_batchnorm" in config else False
    datasize = config["datasize"]
    evalsize = datasize if not args.data_eval_size else args.data_eval_size
    batchsize = 512
    steps = args.alphas

    model, loader = load_model_and_data(
      hsizes, dset_name, act_fn, use_batchnorm, batchsize, datasize, True, False
    )
    evalloader = load_data(dset_name, batchsize, evalsize, True, False, False)
    model.cuda()
    outdir = args.outdir
    try:
      os.makedirs(outdir)
    except:
      pass
  
    init_state = torch.load(os.path.join(expdir, config["run_id"], "init.pt"))["model_state"]
    final_state = torch.load(os.path.join(expdir, config["run_id"], "final.pt"))["model_state"]

    if args.linearity_int:
      alphas, _, logits = interp_networks_eval_examples(
        model, init_state, final_state, loader, evalloader, steps, True
      )
      integral = eval_logit_linearity(logits, alphas)
      all_metrics[i]["linearity_integral"] = integral
    
    if args.compute_gl:
      alphas = np.linspace(0, 1, steps, endpoint=True)
      if not model.use_batchnorm:
        # This version is quicker
        avg_gl = compute_avg_gauss_length(model, init_state, final_state, np.linspace(0, 1, steps, endpoint=True), evalloader)
      else:
        # Slower but handles batch norm correctly
        avg_gl = compute_avg_gauss_length_bn(model, init_state, final_state, np.linspace(0, 1, steps, endpoint=True), loader, evalloader, bn_warm_steps=1)
      all_metrics[i]["gauss_length"] = avg_gl
  
  save_summary_stats(all_configs, all_metrics, outdir)
  save_monotonicity_metrics(all_configs, all_metrics, outdir)


if __name__ == "__main__":
  process_experiments(args.expdir, args.outdir)