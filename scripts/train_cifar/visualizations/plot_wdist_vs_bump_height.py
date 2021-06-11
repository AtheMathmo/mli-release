import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import argparse
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

colors = sns.color_palette()
plt.style.use('seaborn')

parser = argparse.ArgumentParser()
parser.add_argument("summary_path")
parser.add_argument("--thresh", type=float, default=0.1)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

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
            continue
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
    ax.set_ylabel(r"$\min \Delta$", fontsize=14)
    ax.set_title(figtitle, fontsize=14)
    
    mscatter(np.log(weight_dists), bump_height, ax=ax, alpha=0.7, color = color, m=markers)
    plt.tight_layout()
    
    path = os.path.join(os.path.dirname(args.summary_path), figname)
    plt.savefig(path)

    if args.show:
        plt.show()

def main(summary_path, threshold=0.1):
    with open(summary_path) as f:
        data = json.load(f)
    # plot_weight_vs_bump(data)
    plot_weight_vs_bump(
        data, args.thresh, figname='dist_vs_delta.pdf',
        figtitle="Non-monotonicity against weight distance (CIFAR-100)",
        normalized=True
    )
    plot_weight_vs_bump(
        data, args.thresh, figname='dist_vs_delta_adam.png',
        figtitle="Non-monotonicity against weight distance (Adam)",
        normalized=True,
        conf_filter=lambda c : c['optim_name'] == 'adam'
    )
    plot_weight_vs_bump(
        data, args.thresh, figname='dist_vs_delta_sgd.png',
        figtitle="Non-monotonicity against weight distance (SGD)",
        normalized=True,
        conf_filter=lambda c : c['optim_name'] == 'sgd'
    )
    # plot_weight_vs_bump(data, figname='dist_vs_delta_no_bn.png', conf_filter=lambda c : not c['use_batchnorm'])


if __name__ == '__main__':
    main(args.summary_path, args.thresh)