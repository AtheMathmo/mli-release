import os
import numpy as np

from .logit_viz import plot_logits

import mli.metrics as metrics_utils
from .utils import get_run_stats
import argparse


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("expdir")
  parser.add_argument("outdir")
  parser.add_argument("-t", "--thresh", type=float, default=0.1)
  parser.add_argument("-p", "--plot_amount", type=int, default=100)
  parser.add_argument("-a", "--alphas", type=int, default=50)
  args = parser.parse_args()

  all_configs, all_metrics = get_run_stats(args.expdir)
  outdir = args.outdir
  for i in range(len(all_configs)):
    metrics = all_metrics[i]
    alphas = metrics['interpolation.alpha']['values']
    losses = metrics['interpolation.loss']['values']
    if losses[-1] > args.thresh:
      # Only process models with decent final loss
      continue
    _, heights = metrics_utils.eval_monotonic(alphas, losses)
    mono = True
    # Check for non-monotonic
    if len(heights) > 0:
      max_D = np.max(heights)
      if max_D > 0.01:
        mono = False

    rundir = os.path.join(args.expdir, all_configs[i]['run_id'])
    runout = os.path.join(outdir, 'mono' if mono else 'nonmono', all_configs[i]['run_id'])
    os.makedirs(runout, exist_ok=True)
    plot_logits(rundir, args.alphas, runout, args.plot_amount)
