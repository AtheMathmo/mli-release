import os
import itertools
import argparse
import json
import copy

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib.lines import Line2D
import tqdm

from mli.models import FCNet, interpolate_state, get_activation_function
from mli.data import load_data

from .utils import *

parser = argparse.ArgumentParser()
parser.add_argument("rundir")
parser.add_argument("outdir")
parser.add_argument("-d", "--data_eval_size", type=int, default=None)
parser.add_argument("-a", "--alphas", type=int, default=50)
parser.add_argument("--random_states", type=int, default=10)
args = parser.parse_args()


def permute_state(state_dict, permutation, flipperm=None, cuda=True):
  """Permute the weights in the state dictionary, keeping the network function the same.

  Args:

    state_dict: Network state dictionary.
    permutation: A list of row indices representing a permutation

  NOTE: Currently assumes two linear layers only
  """
  perm_mat = torch.zeros(len(permutation), len(permutation))
  for i, p in enumerate(permutation):
    perm_mat[i,p] = 1.0
  if flipperm is not None:
    perm_mat = perm_mat * flipperm
  if cuda:
    perm_mat = perm_mat.cuda()

  left = False
  layers = 0
  flipped = 0
  for key in state_dict:
    layers += 1
    if 'weight' in key:
      # Flip the order of the permutation
      left = not left
      flipped += 1
    if flipped >= 3:
      continue
    if left:
      state_dict[key] = torch.matmul(perm_mat, state_dict[key])
    else:
      state_dict[key] = torch.matmul(state_dict[key], perm_mat.t())
  return state_dict


if __name__ == '__main__':
  run_states = get_run_model_states(args.rundir)
  config = run_states['config']
  hsizes = config['hsizes']
  dset_name = config['dset_name']
  act_fn = get_activation_function(config['act_fn'])
  use_batchnorm = config['use_batchnorm'] if 'use_batchnorm' in config else False
  datasize = config['datasize']
  evalsize = datasize if not args.data_eval_size else args.data_eval_size
  batchsize = 512
  steps = args.alphas

  model, loader = load_model_and_data(
    hsizes, dset_name, act_fn, use_batchnorm, batchsize, datasize, True, False
  )
  evalloader = load_data(dset_name, batchsize, evalsize, True, False, False)
  model.cuda()
  outdir = args.outdir
  try:
    os.makedirs(outdir)
  except:
    pass
  
  init_state = run_states['init_state']
  final_state = run_states['final_state']
  orig_alphas, metrics = interp_networks(model, init_state, final_state, loader, [evalloader], args.alphas, True)
  orig_losses = np.array(metrics[0]['loss'])
  fig, ax = plt.subplots(figsize=(6,3))
  ax.plot(orig_alphas, orig_losses, c='r', alpha=1)

  losses = []
  alphas = None
  for i in range(args.random_states):
    final_state_copy = copy.deepcopy(final_state)
    permutation = np.random.permutation(hsizes[0])
    permuted_state = permute_state(final_state_copy, permutation, None, True)

    alphas, metrics = interp_networks(model, init_state, permuted_state, loader, [evalloader], args.alphas, True)
    losses.append(metrics[0]['loss'])
  losses = np.array(losses)
  mean_l = np.mean(losses, 0)
  std_l = np.std(losses, 0)
  ax.plot(alphas, mean_l, c='b', ls='--')
  ax.fill_between(alphas, mean_l - std_l, mean_l + std_l, facecolor='b', alpha=0.6)
  ax.set_xlim(0,1)
  ax.set_ylim(ymin=0)
  ax.set_title('Interpolating from initialization to permuted final solution', fontsize=16)
  ax.set_xlabel(r"$\alpha$", size=14)
  ax.set_ylabel("Loss", size=14)
  legend_handles=[
      Line2D([0], [0], color='b', ls='--', label='Permuted final'),
      Line2D([0], [0], color='r', label='Original'),
  ]
  ax.legend(handles=legend_handles, loc='upper right', fontsize=14)
  filepath = os.path.join(outdir, 'permute_final.png')
  plt.tight_layout()
  plt.savefig(filepath)

  plt.close()
  fig, ax = plt.subplots(figsize=(6,3))
  ax.plot(orig_alphas, orig_losses, c='r', alpha=1)
  losses = []
  for i in range(args.random_states):
    init_state_copy = copy.deepcopy(init_state)
    permutation = np.random.permutation(hsizes[0])
    permuted_state = permute_state(init_state_copy, permutation, None, True)
    alphas, metrics = interp_networks(model, permuted_state, final_state, loader, [evalloader], args.alphas, True)
    losses.append(metrics[0]['loss'])
  losses = np.array(losses)
  mean_l = np.mean(losses, 0)
  std_l = np.std(losses, 0)
  ax.plot(alphas, mean_l, c='b', ls='--')
  ax.fill_between(alphas, mean_l - std_l, mean_l + std_l, facecolor='b', alpha=0.6)

  ax.set_xlim(0,1)
  ax.set_ylim(ymin=0)
  ax.set_title('Interpolating from permuted initialization to final solution', fontsize=16)
  ax.set_xlabel(r"$\alpha$", size=14)
  ax.set_ylabel("Loss", size=14)
  legend_handles=[
      Line2D([0], [0], color='b', ls='--', label='Permuted init'),
      Line2D([0], [0], color='r', label='Original'),
  ]
  ax.legend(handles=legend_handles, loc='upper right', fontsize=14)
  filepath = os.path.join(outdir, 'permute_init.png')
  plt.tight_layout()
  plt.savefig(filepath)
  plt.close()
