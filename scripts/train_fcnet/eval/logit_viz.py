import argparse


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
plt.style.use('seaborn')
import tqdm

from sklearn.decomposition import PCA

from mli.models import interpolate_state
from mli.data import load_data

from mli_eval.plotting.logits import *
from mli_eval.plotting.interp import plot_interp

from .utils import *


def plot_logits(rundir, alpha_steps, outdir, plot_amount):
  run_states = get_run_model_states(rundir)
  config = run_states['config']
  hsizes = config['hsizes']
  dset_name = config['dset_name']
  act_fn = get_activation_function(config['act_fn'])
  use_batchnorm = config['use_batchnorm'] if 'use_batchnorm' in config else False
  batchsize = 512
  datasize = config['datasize']
  steps = alpha_steps

  model, loader = load_model_and_data(
    hsizes, dset_name, act_fn, use_batchnorm, batchsize, datasize, True, False
  )
  model.cuda()
  outdir = outdir
  try:
    os.makedirs(outdir)
  except:
    pass
  
  init_state = run_states['init_state']
  final_state = run_states['final_state']
  plot_interp(config, run_states['metrics'], outdir)

  eval_loader = load_data(dset_name,  min(plot_amount, 128), plot_amount, True, False, False)
  targets = np.array(eval_loader.dataset.targets)

  alpha_path = os.path.join(outdir, 'alphas.npy')
  logits_path = os.path.join(outdir, 'logits.npy')
  targets_path = os.path.join(outdir, 'targets.npy')
  np.save(targets_path, targets)

  if os.path.isfile(alpha_path) and os.path.isfile(logits_path):
    alphas = np.load(alpha_path)
    logits = np.load(logits_path)
  else:
    alphas, _, logits = interp_networks_eval_examples(model, init_state, final_state, loader, eval_loader, steps, True)
    np.save(alpha_path, alphas)
    np.save(logits_path, logits)
  
  subdir = os.path.join(outdir, "random_ev_logits")
  try:
    os.makedirs(subdir)
  except:
    pass

  plot_random_example_logit_dim_pairs(alphas, logits, subdir)

  subdir = os.path.join(outdir, "projected_logits")
  try:
    os.makedirs(subdir)
  except:
    pass

  projected_logits = pca_project_logits(logits, 2)
  plot_2dprojected_per_class(projected_logits, targets, "PCA", subdir)

  projected_logits = pca_project_logits(logits, 3)
  plot_3dprojected_per_class(projected_logits, targets, "PCA", subdir)

  projected_logits = random_project_logits(logits, 2)
  plot_2dprojected_per_class(projected_logits, targets, "Random Proj", subdir)

  projected_logits = random_project_logits(logits, 3)
  plot_3dprojected_per_class(projected_logits, targets, "Random Proj", subdir)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("rundir")
  parser.add_argument("outdir")
  parser.add_argument("-p", "--plot_amount", type=int, default=100)
  parser.add_argument("-a", "--alphas", type=int, default=50)
  args = parser.parse_args()
  plot_logits(args.rundir, args.alphas, args.outdir, args.plot_amount)
