import os
import itertools
import argparse
import json
import copy
import functools

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import tqdm

from sklearn.decomposition import PCA





def pca_project_logits(logits, dim=2):
  pca = PCA(dim)
  flat_logits = logits.reshape((-1, logits.shape[-1]))
  z = pca.fit_transform(flat_logits)
  return z.reshape((logits.shape[0], logits.shape[1], dim))


def random_project_logits(logits, dim=2):
  indim = logits.shape[-1]
  flat_logits = logits.reshape((-1, indim))
  A = np.random.randn(dim, indim)
  A = A / (A * A).sum(-1, keepdims=True)
  z = np.matmul(flat_logits, A.T)
  return z.reshape((logits.shape[0], logits.shape[1], dim))


def subplot(ax, proj_logits, targets, title=None):
  colormap = plt.get_cmap('tab10')
  targets_max = targets.max()
  for i in range(targets_max):
    plogits_i = proj_logits[:, targets == i, :]
    for j in range(plogits_i.shape[1]):
      ax.plot(
        plogits_i[:,j,0], plogits_i[:,j,1],
        marker='o', markersize=2,
        c=colormap(i), alpha = 0.4)
  if title:
    ax.set_title(title)
  

def plot_2dprojected_per_class(proj_logits1, targets1, proj_logits2, targets2, proj_name, outdir, subdir='projected_logits'):
  savedir = os.path.join(outdir, subdir)

  try:
    os.makedirs(savedir)
  except:
    pass
  fig, axs = plt.subplots(figsize=(10,5), nrows=1, ncols=2)
  subplot(axs[0], proj_logits1, targets1)
  subplot(axs[1], proj_logits2, targets2)
  plt.suptitle(
    "2D logit trajectories projected via {} (colored by class)".format(proj_name),
    fontsize=26
  )
  
  filename = "2D_{}_logit_paths.pdf".format(proj_name)
  plt.tight_layout()
  plt.savefig(os.path.join(savedir, filename))
  plt.close()

def load_plot_data(outdir):
  alpha_path = os.path.join(outdir, 'alphas.npy')
  logits_path = os.path.join(outdir, 'logits.npy')
  targets_path = os.path.join(outdir, 'targets.npy')
  return np.load(alpha_path), np.load(logits_path), np.load(targets_path)

def plot_logit_compare(outdir1, outdir2, plotdir):
  alphas1, logits1, targets1 = load_plot_data(outdir1)
  alphas2, logits2, targets2 = load_plot_data(outdir2)

  projected_logits1 = pca_project_logits(logits1, 2)
  projected_logits2 = pca_project_logits(logits2, 2)
  plot_2dprojected_per_class(projected_logits1, targets1, projected_logits2, targets2, "PCA", plotdir)

  rprojected_logits1 = random_project_logits(logits1, 2)
  rprojected_logits2 = random_project_logits(logits2, 2)
  plot_2dprojected_per_class(rprojected_logits1, targets1, rprojected_logits2, targets2, "Random Proj", plotdir)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("outdir1")
  parser.add_argument("outdir2")
  parser.add_argument("plotdir")
  args = parser.parse_args()
  plot_logit_compare(args.outdir1, args.outdir2, args.plotdir)
