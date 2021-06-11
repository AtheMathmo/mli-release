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

def plot_gl_vs_bump(
    data, threshold=0.1, figname="gl_vs_delta.png",
    figtitle="Non-monotonicity against Gauss length",
    conf_filter=None):
    gauss_lengths = []
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
            gauss_lengths.append(run['gauss_length'])
            bump_height.append(bump)
            color.append(colors[1] if bump > 0 else colors[0])
    
    fig, ax = plt.subplots(figsize=(6,3))
    ax.set_xlabel(r"$\log$ Gauss length", fontsize=14)
    ax.set_ylabel(r"Max $\Delta$", fontsize=14)
    ax.set_title(figtitle, fontsize=16)
    
    mscatter(np.log(gauss_lengths), bump_height, ax=ax, alpha=0.9 if bump > 0 else 0.5, color = color, m=markers)
    plt.tight_layout()
    legend_elements = [
        Line2D([0], [0], marker='o', color=(1,1,1,0), markerfacecolor='b', label='No BatchNorm',markersize=12),
        Line2D([0], [0], marker='s', color=(1,1,1,0), markerfacecolor='b', label='   BatchNorm',markersize=12),
    ]

    ax.legend(handles=legend_elements, loc='upper right')
    if args.show:
        plt.show()
    else:
        path = os.path.join(os.path.dirname(args.summary_path), figname)
        plt.savefig(path)

def main(summary_path, threshold=0.1):
    with open(summary_path) as f:
        data = json.load(f)
    plot_gl_vs_bump(
        data, figname='gl_vs_delta.pdf',
        figtitle="Non-monotonicity against Gauss length",
        conf_filter=lambda c : not c['use_batchnorm']
    )


if __name__ == '__main__':
    main(args.summary_path, args.thresh)