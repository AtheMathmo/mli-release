import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import argparse
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from mli_eval.plotting.scatter import mscatter

colors = sns.color_palette()
plt.style.use('seaborn')


parser = argparse.ArgumentParser()
parser.add_argument("summary_path")
parser.add_argument("--thresh", type=float, default=0.1)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()


def plot_weight_vs_bump(
    data, threshold=0.1, figname="dist_vs_delta.png",
    figtitle="Non-monotonicity against weight distance",
    normalized=False, conf_filter=None):
    weight_dists = []
    bump_height = []
    loss = []
    color = []
    markers = []
    
    for run in data:
        marker = "o"
        if conf_filter is not None and not conf_filter(run['config']):
            marker = "s"
        min_loss = run['min_train_loss']
        if min_loss < threshold:
            markers.append(marker)
            loss.append(min_loss)
            bump = run['max_bump']
            weight_dists.append(run['normed_weight_dist'] if normalized else run['weight_dist'])
            bump_height.append(bump)
            color.append(colors[1] if bump > 0 else colors[0])
    
    fig, ax = plt.subplots(figsize=(6,3))
    # plt.title(r"Max $\Delta$ vs distance from initialization", fontsize=18)
    xlabel = r"$\log \Vert \theta_T - \theta_0 \Vert_2$"
    if normalized:
        xlabel += r" - $\log \Vert \theta_0 \Vert_2$"
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(r"Max $\Delta$", fontsize=14)
    ax.set_title(figtitle, fontsize=16)
    
    mscatter(np.log(weight_dists), bump_height, ax=ax, alpha=0.7, color = color, m=markers)
    plt.tight_layout()

    if args.show:
        plt.show()
    else:
        path = os.path.join(os.path.dirname(args.summary_path), figname)
        plt.savefig(path)

def main(summary_path, threshold=0.1):
    with open(summary_path) as f:
        data = json.load(f)
    # plot_weight_vs_bump(data)
    plot_weight_vs_bump(
        data, figname='dist_vs_delta.pdf',
        threshold=threshold,
        figtitle="Non-monotonicity against weight distance",
        normalized=False
    )
    plot_weight_vs_bump(
        data, figname='dist_vs_delta_normalized.pdf',
        threshold=threshold,
        figtitle="Non-monotonicity against normalized weight distance",
        normalized=True
    )
    # plot_weight_vs_bump(data, figname='dist_vs_delta_no_bn.png', conf_filter=lambda c : not c['use_batchnorm'])


if __name__ == '__main__':
    main(args.summary_path, args.thresh)
