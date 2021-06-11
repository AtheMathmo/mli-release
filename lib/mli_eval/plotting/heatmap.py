import os

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np


def save_delta_heatmap(
    monotonicity_summary, outdir,
    map1, map2,
    map1_label, map2_label,
    loss_thresh,
    figpath="delta_heatmap.pdf",
    cbar_range=None,
    conf_filter=None
  ):
  results = {}
  map2_set = set()

  for i in range(len(monotonicity_summary)):
    config = monotonicity_summary[i]["config"] # Contains the run config
    final_loss = monotonicity_summary[i]["final_train_loss"]
    delta = monotonicity_summary[i]["max_bump"] # The max Delta in our paper

    if conf_filter is not None and not conf_filter(config):
        continue
    if final_loss > loss_thresh:
        continue

    # Our axes are learning rate and hidden size of first layer
    x = map1(config)
    y = map2(config)
    map2_set.add(y)

    if x not in results:
      results[x] = {}
    if y not in results[x]:
      results[x][y] = { "count": 0, "delta": 0}
    if delta > 0:
      results[x][y]["count"] += 1
      results[x][y]["delta"] += delta
  # Extract the list of LRs and hidden sizes
  # Sort them
  xs = list(results.keys())
  xs.sort(reverse=True)
  x_total = len(xs)
  ys = list(map2_set)
  ys.sort()
  y_total = len(ys)

  # Fill the heatmap with the Delta data
  heatmap = np.zeros((x_total, y_total, 4))
  cmap = plt.cm.get_cmap("inferno")
  if cbar_range is not None:
    minD = cbar_range[0]
    maxD = cbar_range[1]
  else:
    minD = 1000
    maxD = 0
    # Compute the delta range
    for i in range(x_total):
      for j in range(y_total):
        if ys[j] in results[xs[i]] and results[xs[i]][ys[j]]["count"] > 0:
          avg_D = results[xs[i]][ys[j]]["delta"] / results[xs[i]][ys[j]]["count"]
          if avg_D < minD:
            minD = avg_D
          if avg_D > maxD:
            maxD = avg_D
  for i in range(x_total):
    for j in range(y_total):
      if ys[j] in results[xs[i]] and results[xs[i]][ys[j]]["count"] > 0:
        avg_D = results[xs[i]][ys[j]]["delta"] / results[xs[i]][ys[j]]["count"]
        normed_D = (avg_D - minD) / (maxD - minD)
        heatmap[i,j] = cmap(normed_D)
      else:
        heatmap[i,j] = (0,1,1,1)
  # Create the figure
  fig, ax = plt.subplots(figsize=(6 * y_total / x_total, 6))
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.2)

  # Removes the major grid from the plot
  ax.grid(b=None)
  im = ax.imshow(heatmap)

  norm = mpl.colors.Normalize(vmin=minD, vmax=maxD)
  cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
  cbar.set_label(r"$\Delta_{\min}$", fontsize=14)
  # cbar = ax.figure.colorbar(im, ax=ax)
  # cbar.ax.set_ylabel(r"$\Delta_{\min}$", rotation=-90, va="bottom", fontsize=14)

  # We"ll put the labels along the major ticks
  ax.set_xticks(np.arange(y_total))
  ax.set_yticks(np.arange(x_total))
  ax.set_xticklabels(ys)
  ax.set_yticklabels(xs)

  ax.set_ylabel(map1_label, fontsize=14)
  ax.set_xlabel(map2_label, fontsize=14)
  
  # This produces the minor grid, isolating each entry with wrapping
  ax.set_xticks(np.arange(y_total+1)-.5, minor=True)
  ax.set_yticks(np.arange(x_total+1)-.5, minor=True)
  ax.grid(which="minor", color="w", linestyle="-", linewidth=3)

  # plt.tight_layout()
  fpath = os.path.join(outdir, figpath)
  plt.savefig(fpath)