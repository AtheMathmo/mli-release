import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import argparse
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from sklearn.linear_model import LinearRegression

colors = sns.color_palette()
plt.style.use('seaborn')

parser = argparse.ArgumentParser()
parser.add_argument("expdir")
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

def plot_gl_vs_wdist(
    data, gls, threshold=0.1, figname="gl_vs_wdist.png",
    figtitle="Distance in weight space against Gauss length",
    normalized=False, conf_filter=None):
    gauss_lengths = []
    wdists = []
    loss = []
    markers = []
    
    for run in data:
        if run['run'] not in gls:
            continue
        marker = 'o'
        if conf_filter is not None and not conf_filter(run['config']):
            continue
        min_loss = run['min_train_loss']
        if min_loss < threshold:
            markers.append(marker)
            loss.append(min_loss)
            gauss_lengths.append(gls[run['run']])
            wdists.append(run['normed_weight_dist'] if normalized else run['weight_dist'])
    
    fig, ax = plt.subplots(figsize=(6,3))
    xlabel = r"$\log \Vert \theta_T - \theta_0 \Vert_2$"
    if normalized:
        xlabel += r" - $\log \Vert \theta_0 \Vert_2$"
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(r"$\log$(Gauss length)", fontsize=14)
    ax.set_title(figtitle, fontsize=16)
    
    log_wd = np.log(wdists)
    log_gl = np.log(gauss_lengths)
    mscatter(log_wd, log_gl, ax=ax, alpha=1, color = 'b', m=markers)
    plt.tight_layout()
    path = os.path.join(os.path.dirname(args.summary_path), figname)
    plt.savefig(path)
    
    reg = LinearRegression().fit(log_wd.reshape(-1, 1), log_gl.reshape(-1, 1))
    r2 = reg.score(log_wd.reshape(-1, 1), log_gl.reshape(-1, 1))
    print("Goodness of fit: {}".format(r2))
    print("Slope: {}".format(reg.coef_))
    if args.show:
        plt.show()


def get_gauss_lengths(expdir):
  gauss_lens = {}
  alldirs = os.listdir(expdir)
  for rundir in alldirs:
    # Sacred saves all source code
    if rundir == '_sources':
      continue
    glpath = os.path.join(expdir, rundir, 'gl.txt')
    if not os.path.isfile(glpath):
      continue
    gl = np.loadtxt(glpath)
    if np.isnan(gl):
      continue
    gauss_lens[os.path.split(rundir)[1]] = gl
  return gauss_lens


def main(expdir, summary_path, threshold=0.1):
    with open(summary_path) as f:
        data = json.load(f)
    gls = get_gauss_lengths(expdir)
    plot_gl_vs_wdist(
        data, gls, figname='gl_vs_wdist.pdf',
        figtitle=r"CIFAR-10 Gauss length vs weight distance",
        threshold=threshold,
        normalized=True
    )
    plot_gl_vs_wdist(
        data, gls, figname='gl_vs_wdist_unnorm.pdf',
        figtitle=r"CIFAR-10 Gauss length vs weight distance",
        threshold=threshold,
        normalized=False
    )


if __name__ == '__main__':
    main(args.expdir, args.summary_path, args.thresh)