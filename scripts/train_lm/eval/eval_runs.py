import argparse

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.style.use("seaborn")

from scripts.train_lm.eval.utils import *
from mli_eval.plotting.interp import plot_all_lm_interp
from mli_eval.processing.experiments import get_monotonicity_metrics, summarize_lm_metrics, get_run_lm_stats


parser = argparse.ArgumentParser()
parser.add_argument("--expdir", default="../runs/mli_lm/")
parser.add_argument("--outdir", default=".")
parser.add_argument("--plot_loss_lb", type=float, default=None)
args = parser.parse_args()


def model_cmap(config):
  model_name = config["model"]
  if "lstm" in model_name:
    return "g"
  else:
    return "purple"


def optim_cmap(config):
  optim_name = config["optimizer"]
  if "sgd" in optim_name:
    return "g"
  else:
    return "purple"


def process_experiments(expdir, outdir):
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  all_configs, all_metrics = get_run_lm_stats(expdir)
  summary = summarize_lm_metrics(
    all_configs, all_metrics,
    metric_summaries=[(np.min, "val.interpolation.loss")]
  )
  with open(os.path.join(outdir, "summary.json"), "w") as f:
    json.dump(summary, f)

  monotonicity_summary = get_monotonicity_metrics(all_configs, all_metrics)
  with open(os.path.join(outdir, "monotonicity_metrics.json"), "w") as f:
    json.dump(monotonicity_summary, f)
  plot_all_lm_interp(
    all_configs, all_metrics, outdir,
    loss_lb=1000., train_prefix="train",
    figname="all_interp_train.pdf",
    figtitle="Linear interpolations for Language Modelling",
    legend_handles=[
        Line2D([0], [0], color="g", label="LSTM"),
        Line2D([0], [0],color="purple", label="Transformer"),
    ],
    color_map=model_cmap
  )

  plot_all_lm_interp(
    all_configs, all_metrics, outdir,
    loss_lb=1000., train_prefix="train",
    figname="all_interp_train_optim.pdf",
    figtitle="Linear interpolations for Language Modelling",
    legend_handles=[
        Line2D([0], [0], color="g", label="SGD"),
        Line2D([0], [0],color="purple", label="Adam"),
    ],
    color_map=optim_cmap
  )

if __name__ == "__main__":
  process_experiments(args.expdir, args.outdir)
