import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
plt.style.use("seaborn")

from sklearn.decomposition import PCA


def plot_random_example_logit_dim_pairs(alphas, logits, outdir, examples=10):
    total_ex = logits.shape[1]
    data_indices = np.random.choice(np.arange(total_ex), examples)

    for idx in data_indices:
        fig = plt.figure(figsize=(10,10), constrained_layout=True)
        gs = fig.add_gridspec(4, 3)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_title("All logits")
        ax1.set_xlim(0, 1)
        im_name = "{}_logits.png".format(str(idx))
        h = logits[:,idx]
        ax1.plot(alphas, h, color="blue")

    h0 = h[:,0]
    for i in range(1,logits.shape[-1]):
        row = 1 + (i - 1) // 3
        col = (i - 1) % 3
        ax = fig.add_subplot(gs[row, col])
        ax.set_title("Logit dims 0 - {}".format(i))
        hi = h[:, i]
        ax.plot(h0, hi, marker="o")
    plt.savefig(os.path.join(outdir, im_name))
    plt.close(fig)


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


def plot_2dprojected_per_class(proj_logits, targets, proj_name, outdir):
  assert proj_logits.shape[-1] == 2, "Projected logits must be 2D"
  colormap = plt.get_cmap("tab10")
  targets_max = targets.max()
  plt.figure(figsize=(10,10))
  plt.title(
    "2D Logit trajectories projected via {}\nColored by class".format(proj_name),
    fontsize=18
  )
  plot_amount = proj_logits.shape[0]
  for i in range(targets_max):
      plogits_i = proj_logits[:, targets == i, :]
      for j in range(plogits_i.shape[1]):
          plt.plot(
            plogits_i[:,j,0], plogits_i[:,j,1],
            marker="o", markersize=2,
            c=colormap(i), alpha = 0.4)
  filename = "2D_{}_logit_paths_{}.png".format(proj_name, str(plot_amount))
  plt.savefig(os.path.join(outdir, filename))
  plt.close()


def plot_3dprojected_per_class(proj_logits, targets, proj_name, outdir):
    assert proj_logits.shape[-1] == 3, "Projected logits must be 3D"
    colormap = plt.get_cmap("tab10")
    targets_max = targets.max()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plt.title(
      "3D Logit trajectories projected via {}\nColored by class".format(proj_name),
      fontsize=18
    )
    plot_amount = proj_logits.shape[0]
    for i in range(targets_max):
        plogits_i = proj_logits[:, targets == i, :]
        for j in range(plogits_i.shape[1]):
            ax.plot(
              plogits_i[:,j,0], plogits_i[:,j,1], plogits_i[:,j,2],
              marker="o", markersize=2,
              c=colormap(i), alpha = 0.4)
    filename = "3D_{}_logit_paths_{}.png".format(proj_name, str(plot_amount))
    plt.savefig(os.path.join(outdir, filename))
    plt.close()
