import argparse
import os
import json

import matplotlib.pyplot as plt
plt.style.use("seaborn")
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mli.metrics as metrics_utils
from mli_eval.plotting.interp import plot_all_interp, plot_interp
from mli_eval.plotting.heatmap import save_delta_heatmap
from mli_eval.processing.experiments import summarize_metrics, get_monotonicity_metrics

from .utils import *

parser = argparse.ArgumentParser()
parser.add_argument("expdir")
parser.add_argument("outdir")
args = parser.parse_args()


def process_experiments(expdir, outdir):
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  all_configs, all_metrics = get_run_stats(expdir)
  summary = summarize_metrics(
    all_configs, all_metrics,
    metric_summaries=[(np.min, "train.loss"), (np.max, "train.acc")]
  )
  with open(os.path.join(outdir, "summary.json"), "w") as f:
    json.dump(summary, f)

  monotonicity_summary = get_monotonicity_metrics(all_configs, all_metrics)
  with open(os.path.join(outdir, "monotonicity_metrics.json"), "w") as f:
    json.dump(monotonicity_summary, f)
  save_delta_heatmap(
    monotonicity_summary, outdir,
    lambda c: c["lr"], lambda c: len(c["hsizes"]),
    "Learning rate", "Depth",
    0.5, "delta_heatmap_lr_depth.pdf",
    cbar_range=(0,18),
    conf_filter = lambda c : c["hsizes"][1] == 1024
  )
  save_delta_heatmap(
    monotonicity_summary, outdir,
    lambda c: c["lr"], lambda c: c["hsizes"][1],
    "Learning rate", "Hidden size",
    0.5, "delta_heatmap_lr_hsize.pdf",
    cbar_range=(0,18),
    conf_filter = lambda c : len(c["hsizes"]) == 3
  )
  plot_all_interp(
    all_configs, all_metrics, outdir,
    figname="all_interp_mnist.pdf",
    figtitle="Linear interpolations of MNIST networks",
    legend_handles=[
        Line2D([0], [0], color="b", label="No BatchNorm"),
        Line2D([0], [0],color="r", label="BatchNorm"),
    ],
    conf_filter=lambda c : c["dset_name"] == "mnist",
    color_map=lambda c : "r" if c["use_batchnorm"] else "b"
  )
  plot_all_interp(
    all_configs, all_metrics, outdir,
    figname="all_interp_fmnist.pdf",
    figtitle="Linear interpolations of FashionMNIST networks",
    legend_handles=[
        Line2D([0], [0], color="b", label="No BatchNorm"),
        Line2D([0], [0],color="r", label="BatchNorm"),
    ],
    conf_filter=lambda c : c["dset_name"] == "fashionmnist",
    color_map=lambda c : "r" if c["use_batchnorm"] else "b"
  )

  for i in range(len(all_configs)):
    plot_interp(all_configs[i], all_metrics[i], outdir)


if __name__ == "__main__":
  process_experiments(args.expdir, args.outdir)
