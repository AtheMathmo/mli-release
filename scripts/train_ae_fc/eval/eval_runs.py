import argparse
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from matplotlib.lines import Line2D
from matplotlib import cm

from mli_eval.plotting.interp import plot_interp, plot_all_interp, plot_all_interp_comparison
from mli_eval.processing.experiments import get_monotonicity_metrics, summarize_metrics

from .utils import *


parser = argparse.ArgumentParser()
parser.add_argument("expdir")
parser.add_argument("outdir")
parser.add_argument("--thresh", type=float, default=30)
args = parser.parse_args()


def get_lr_bounds(all_confs, all_metrics, lr_bound=None, conf_filter=None):
    lr_min = 10000
    lr_max = 0
    seen = 0
    for i in range(len(all_confs)):
        c = all_confs[i]
        m = all_metrics[i]
        if conf_filter is not None and not conf_filter(c):
            continue
        if lr_bound is not None and m["train.loss"]["values"][-1] < lr_bound:
            continue
        seen += 1
        lr = c["lr"]
        if lr < lr_min:
            lr_min = lr
        if lr > lr_max:
            lr_max = lr
    if seen == 0:
        raise ValueError("Empty config list (maybe from filtering)")
    return lr_min, lr_max


def lr_cmap(config, lr_min=0, lr_max=1):
    lr = config["lr"]
    cmap = cm.get_cmap("inferno")
    cbar_norm = mpl.colors.LogNorm(vmin=lr_min, vmax=lr_max)
    return cmap(cbar_norm(lr))


def hsize_filter(config, h=None):
    if h:
        return config["hsizes"][1] == h
    else:
        return True

def process_experiments(expdir, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    all_configs, all_metrics = get_run_stats(expdir)
    summary = summarize_metrics(
        all_configs, all_metrics, metric_summaries=[(np.min, "train.loss")]
    )
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f)

    monotonicity_summary = get_monotonicity_metrics(all_configs, all_metrics)
    with open(os.path.join(outdir, "monotonicity_metrics.json"), "w") as f:
        json.dump(monotonicity_summary, f)

    plot_all_interp(
        all_configs, all_metrics, outdir,
        figname="all_interp_mnist.png",
        figtitle="Linear interpolations of MNIST autoencoders",
        loss_lb=args.thresh,
    )

    # Plot all
    sgd_lr_min, sgd_lr_max = get_lr_bounds(
        all_configs,
        all_metrics,
        args.thresh,
        lambda c: c["optim_name"] == "sgd"
    )

    plot_all_interp(
        all_configs, all_metrics, outdir,
        figname="ae_mnist_interp_sgd.pdf",
        figtitle="Linear interpolations of MNIST autoencoders (SGD)",
        loss_lb=args.thresh,
        conf_filter=lambda c: c["optim_name"] == "sgd",
        color_map=lambda c: "b"
        # color_map=partial(lr_cmap, lr_min=sgd_lr_min, lr_max=sgd_lr_max),
        # colorbar=cbar
    )

    adam_lr_min, adam_lr_max = get_lr_bounds(
        all_configs,
        all_metrics,
        args.thresh,
        lambda c: c["optim_name"] == "adam"
    )
    cbar_norm = mpl.colors.LogNorm(vmin=adam_lr_min, vmax=adam_lr_max)
    cbar = partial(mpl.colorbar.ColorbarBase, norm=cbar_norm, cmap=cm.get_cmap("inferno"))
    plot_all_interp(
        all_configs, all_metrics, outdir,
        figname="ae_mnist_interp_adam.pdf",
        figtitle="Linear interpolations of MNIST autoencoders (Adam)",
        loss_lb=args.thresh,
        conf_filter=lambda c: c["optim_name"] == "adam",
        color_map=lambda c: "b"
        # color_map=partial(lr_cmap, lr_min=adam_lr_min, lr_max=adam_lr_max),
        # colorbar=cbar
    )

    plot_all_interp_comparison(
        all_configs, all_metrics, outdir,
        lambda c: c["optim_name"] == "sgd",
        args.thresh,
        figtitle="Linear interpolations of MNIST autoencoders",
        figname="ae_mnist_adam_sgd.pdf",
        color_map=lambda c: "b",
        subtitles=("SGD", "Adam")
    )

    for h in [100, 50, 25, 10, 5, 2, 1]:
        plot_all_interp(
        all_configs, all_metrics, outdir,
            figname="ae_mnist_interp_h_{}.png".format(h),
            figtitle="Linear interpolations of MNIST autoencoders (H={})".format(h),
            loss_lb=args.thresh,
            conf_filter=partial(hsize_filter, h=h),
            color_map=lambda c: "b"
        )

    for i in range(len(all_configs)):
        plot_interp(all_configs[i], all_metrics[i], outdir)



if __name__ == "__main__":
    process_experiments(args.expdir, args.outdir)
