import os

import matplotlib.pyplot as plt
plt.style.use('seaborn')
import numpy as np

import mli.metrics as metrics_utils


def plot_all_interp(
  configs, metrics, outdir, train_prefix="train",
  loss_lb=1, figname="all_interp.png",
  figtitle="Monotonic linear interpolations",
  legend_handles=None,
  conf_filter=None,
  color_map=None):
  _, ax = plt.subplots(figsize=(6,3))
  for i in range(len(metrics)):
    if conf_filter is not None and not conf_filter(configs[i]):
      continue
    c = None
    if color_map is not None:
      c = color_map(configs[i])
    final_loss = metrics[i]['train.loss']['values'][-1]
    if loss_lb is not None and final_loss > loss_lb:
      continue
    alpha_steps = metrics[i]['{}.interpolation.alpha'.format(train_prefix)]['values']
    interp_losses = metrics[i]['{}.interpolation.loss'.format(train_prefix)]['values']
    _, heights = metrics_utils.eval_monotonic(alpha_steps, interp_losses)
    if len(heights) > 0:
      alpha = 0.5
    else:
      alpha = 0.2
    ax.plot(alpha_steps, interp_losses, alpha=alpha, c=c)
  fpath = os.path.join(outdir, figname)
  if legend_handles:
    ax.legend(handles=legend_handles, loc='lower left')
  ax.set_xlim(0, 1)
  ax.set_ylim(ymin=0, ymax=4)
  ax.set_title(figtitle, fontsize=16)
  ax.set_xlabel(r"$\alpha$", size=14)
  ax.set_ylabel("Loss", size=14)
  plt.tight_layout()
  plt.savefig(fpath)
  plt.clf()
  plt.close()


def plot_all_lm_interp(
  configs, metrics, outdir, train_prefix="train",
  loss_lb=1, figname="all_interp.png",
  figtitle="Monotonic linear interpolations",
  legend_handles=None,
  conf_filter=None,
  color_map=None):
  _, ax = plt.subplots(figsize=(6,3))
  for i in range(len(metrics)):
    if conf_filter is not None and not conf_filter(configs[i]):
      continue
    c = None
    if color_map is not None:
      c = color_map(configs[i])
    try:
      final_loss = metrics[i]['train.ppl']['values'][-1]
      # final_loss = metrics[i]["train.interpolation.loss"]["values"][-1]
      if loss_lb is not None and final_loss > loss_lb:
        continue
      else:
        alpha_steps = metrics[i]['{}.interpolation.alpha'.format(train_prefix)]['values']
        interp_losses = metrics[i]['{}.interpolation.loss'.format(train_prefix)]['values']
        # interp_losses = np.exp(np.array(metrics[i]['{}.interpolation.loss'.format(train_prefix)]['values']))
        _, heights = metrics_utils.eval_monotonic(alpha_steps, interp_losses)
        if len(heights) > 0:
          alpha = 0.5
        else:
          alpha = 0.2
        ax.plot(alpha_steps, interp_losses, alpha=alpha, c=c)
    except:
      pass
  fpath = os.path.join(outdir, figname)
  if legend_handles:
    ax.legend(handles=legend_handles, loc='lower left')
  ax.set_xlim(0, 1)
  # ax.set_ylim(ymin=0, ymax=4)
  ax.set_title(figtitle, fontsize=16)
  ax.set_xlabel(r"$\alpha$", size=14)
  ax.set_ylabel("Loss", size=14)
  plt.tight_layout()
  plt.show()
  plt.clf()
  plt.close()

def plot_interp(config, metrics, outdir, fname_labels=[]):
  epochs = config["epochs"]
  run_id = config["run_id"]
  try:
    train_losses = metrics['train.loss']['values']
    alpha_steps = metrics['train.interpolation.alpha']['values']
    interp_losses = metrics['train.interpolation.loss']['values']
  except:
    return

  fig = plt.figure(figsize=(6,3))
  ax1 = fig.add_subplot(111)
  ax2 = ax1.twiny()

  ax1.plot(alpha_steps, np.array(interp_losses), c='green', ls='--', label='Interpolation Loss')
  ax1.set_xlabel(r"Interpolation ($\alpha$)", size=14)
  ax1.set_ylabel("Loss", size=14)
  ax1.set_xlim(0, 1)

  ax2.plot(np.linspace(0, epochs, len(train_losses)), train_losses, label='Train Loss')
  ax2.set_xlim(0, epochs)
  ax2.set_xlabel("Train epochs", size=14)
  ax2.set_xticks([0, epochs * 0.2, epochs * .4, epochs * .6, epochs * 0.8, epochs])

  handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
  plt.legend(handles, labels, frameon=True, fontsize=18)

  fname = "run_id={}".format(run_id)
  if len(fname_labels) > 0:
    fname += "," + ",".join(["{}={}".format(l, str(config[l])) for l in fname_labels])
  fpath = os.path.join(outdir, fname)
  plt.tight_layout()
  plt.savefig(fpath + ".png")
  plt.savefig(fpath + ".pdf")
  plt.clf()
  plt.close()


def plot_all_interp_comparison(
  configs, metrics, outdir,
  conf_filter, loss_lb=0.1, figname="interp_compare.png",
  figtitle="Monotonic linear interpolations",
  subtitles=None,
  legend_handles=None,
  color_map=None):
  fig, axs = plt.subplots(figsize=(8,3), nrows=1, ncols=2, sharex=True, sharey=True)
  colors = []
  for i in range(len(metrics)):
    c = None
    if color_map is not None:
      c = color_map(configs[i])
    colors.append(c)
    final_loss = metrics[i]['train.loss']['values'][-1]
    if loss_lb is None or final_loss <= loss_lb:
      alpha_steps = metrics[i]['train.interpolation.alpha']['values']
      interp_losses = metrics[i]['train.interpolation.loss']['values']
      if conf_filter(configs[i]):
        axs[0].plot(alpha_steps, interp_losses, alpha=0.4, c=c)
      else:
        axs[1].plot(alpha_steps, interp_losses, alpha=0.4, c=c)
  fpath = os.path.join(outdir, figname)
  if legend_handles:
    axs[1].legend(handles=legend_handles, loc='upper right')
  axs[0].set_xlim(0, 1)
  axs[0].set_ylim(ymin=0)
  axs[0].set_xlabel(r"$\alpha$", size=14)
  axs[1].set_xlabel(r"$\alpha$", size=14)
  axs[0].set_ylabel("Loss", size=14)

  if subtitles is not None:
    axs[0].set_title(subtitles[0])
    axs[1].set_title(subtitles[1])
  plt.suptitle(figtitle, fontsize=16)
  plt.tight_layout()
  plt.savefig(fpath)
  plt.clf()
  plt.close()