import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import argparse
from mli_eval.plotting.scatter import mscatter

from sklearn.linear_model import LinearRegression


colors = sns.color_palette()
plt.style.use("seaborn")


parser = argparse.ArgumentParser()
parser.add_argument("summary_path")
parser.add_argument("--thresh", type=float, default=0.1)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()


def plot_gl_vs_wdist(
    data, threshold=0.1, figname="gl_vs_wdist.png",
    figtitle="Distance in weight space against Gauss length",
    normalized=False, conf_filter=None):
    gauss_lengths = []
    wdists = []
    loss = []
    markers = []
    
    for run in data:
        marker = "o"
        if conf_filter is not None and not conf_filter(run["config"]):
            continue
        min_loss = run["min_train_loss"]
        if min_loss < threshold:
            markers.append(marker)
            loss.append(min_loss)
            gauss_lengths.append(run["gauss_length"])
            wdists.append(run["normed_weight_dist"] if normalized else run["weight_dist"])
    
    fig, ax = plt.subplots(figsize=(6,3))
    xlabel = r"$\log \Vert \theta_T - \theta_0 \Vert_2$"
    if normalized:
        xlabel += r" - $\log \Vert \theta_0 \Vert_2$"
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(r"$\log$(Gauss length)", fontsize=14)
    ax.set_title(figtitle, fontsize=16)
    
    log_wd = np.log(wdists)
    log_gl = np.log(gauss_lengths)
    mscatter(log_wd, log_gl, ax=ax, alpha=1, color = "b", m=markers)
    plt.tight_layout()
    
    path = os.path.join(os.path.dirname(args.summary_path), figname)
    plt.savefig(path)
    if args.show:
        plt.show()
    
    reg = LinearRegression().fit(log_wd.reshape(-1, 1), log_gl.reshape(-1, 1))
    r2 = reg.score(log_wd.reshape(-1, 1), log_gl.reshape(-1, 1))
    print("Goodness of fit: {}".format(r2))
    print("Slope: {}".format(reg.coef_))

def main(summary_path, threshold=0.1):
    with open(summary_path) as f:
        data = json.load(f)
    plot_gl_vs_wdist(
        data, figname="gl_vs_wdist.pdf",
        figtitle=r"Autoencoder Gauss length vs weight distance",
        threshold=threshold,
        normalized=True
    )

    plot_gl_vs_wdist(
        data, figname="gl_vs_wdist_unnorm.pdf",
        figtitle=r"Autoencoder Gauss length vs weight distance",
        threshold=threshold,
        normalized=False
    )


if __name__ == "__main__":
    main(args.summary_path, args.thresh)