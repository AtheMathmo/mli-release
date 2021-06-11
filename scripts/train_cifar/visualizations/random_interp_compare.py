import os
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib.lines import Line2D

def load_plot_data(outdir):
  alpha_path = os.path.join(outdir, 'alphas.npy')
  orig_losses_path = os.path.join(outdir, 'orig_losses.npy')
  rand_losses_path = os.path.join(outdir, 'rand_losses.npy')

  alphas = np.load(alpha_path)
  orig_losses = np.load(orig_losses_path)
  rand_losses = np.load(rand_losses_path)
  return alphas, orig_losses, rand_losses


def plot_data(a1, ol1, rl1, a2, ol2, rl2, savedir):
  m1 = np.mean(rl1, 0)
  s1 = np.std(rl1, 0)

  m2 = np.mean(rl2, 0)
  s2 = np.std(rl2, 0)
  fig, axs = plt.subplots(figsize=(6,4), nrows=2, ncols=1, sharex=True, sharey=True)
  axs[0].set_xlim(0,1)
  axs[0].set_ylim(0, 3.4)
  axs[1].set_xlabel(r"$\alpha$", size=14)
  axs[0].set_ylabel("Loss", size=14)
  axs[1].set_ylabel("Loss", size=14)
  plt.suptitle("Interpolating initializations to final solution (CIFAR-10)", fontsize=18)
  
  axs[0].plot(a1, ol1, c='r', alpha=1)
  axs[0].plot(a1, m1, color='b', ls='--')
  axs[0].fill_between(a1, m1 - s1, m1 + s1, facecolor='b', alpha=0.6)

  axs[1].plot(a2, ol2, c='r', alpha=1)
  axs[1].plot(a2, m2, color='b', ls='--')
  axs[1].fill_between(a2, m2 - s2, m2 + s2, facecolor='b', alpha=0.6)
  legend_handles=[
    Line2D([0], [0], color='b', ls='--', label='Random init'),
    Line2D([0], [0], color='r', label='Original'),
  ]
  axs[1].legend(handles=legend_handles, loc='lower left', fontsize=14)
  plt.tight_layout()
  plt.savefig(os.path.join(savedir, 'rand_init_compare.pdf'))



def plot_interp_compare(outdir1, outdir2, plotdir):
  alphas1, orig_l1, rand_l1 = load_plot_data(outdir1)
  alphas2, orig_l2, rand_l2 = load_plot_data(outdir2)
  plot_data(alphas1, orig_l1, rand_l1, alphas2, orig_l2, rand_l2, plotdir)


  

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("outdir1")
  parser.add_argument("outdir2")
  parser.add_argument("plotdir")
  args = parser.parse_args()
  plot_interp_compare(args.outdir1, args.outdir2, args.plotdir)