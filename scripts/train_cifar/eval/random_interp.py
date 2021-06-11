import os
import itertools
import argparse
import json
import copy

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib.lines import Line2D
import tqdm

from mli.models import get_activation_function
from mli.data import load_data
from .utils import get_model, get_run_model_states, interp_networks, load_model_and_data

parser = argparse.ArgumentParser()
parser.add_argument("rundir")
parser.add_argument("outdir")
parser.add_argument("-d", "--data_eval_size", type=int, default=None)
parser.add_argument("-a", "--alphas", type=int, default=50)
parser.add_argument("--random_states", type=int, default=10)
parser.add_argument("--init_scale", type=float, default=1.0)
parser.add_argument("--show", action='store_true')
args = parser.parse_args()


def random_state_dict(model_state):
    """Returns a new state dictionary filled with random values
    """
    state = {}
    for p_name in model_state:
      # copy the state
      state[p_name] = model_state[p_name].clone().detach()
      # reinitialize if weight or bias
      if 'bias' in p_name:
        # using fanout, as hard to get fanin reliably
        bound = args.init_scale / np.sqrt(state[p_name].shape[0])
        nn.init.uniform_(state[p_name], -bound, bound)
      if 'weight' in p_name:
        nn.init.kaiming_uniform_(state[p_name], a=np.sqrt(5))
        state[p_name] *= args.init_scale
    return state
        
def randomly_perturb_state(model_state, stddev=1):
    state = {}
    for p_name in model_state:
      # copy the state
      state[p_name] = model_state[p_name].clone().detach()
      # Noise the weights and biases
      if 'bias' in p_name or 'weight' in p_name:
        state[p_name] += torch.randn_like(state[p_name]) * stddev
    return state


def compute_interp_data(model, loader, evalloader, init_state, get_rand_state, final_state):
  orig_alphas, metrics = interp_networks(model, init_state, final_state, loader, [evalloader], args.alphas, True)
  orig_losses = np.array(metrics[0]['loss'])
  # From random initialization
  alphas = None
  losses = []
  for _ in range(args.random_states):
    # Get a new random model
    random_state = get_rand_state()
    alphas, metrics = interp_networks(model, random_state, final_state, loader, [evalloader], args.alphas, True)
    losses.append(
      metrics[0]['loss']
    )
  rand_losses = np.array(losses)
  np.save(os.path.join(outdir, 'alphas'), orig_alphas)
  np.save(os.path.join(outdir, 'rand_losses'), rand_losses)
  np.save(os.path.join(outdir, 'orig_losses'), orig_losses)
  return orig_alphas, orig_losses, rand_losses

if __name__ == '__main__':
  run_states = get_run_model_states(args.rundir)
  config = run_states['config']
  model_name = config['model_name']
  num_classes = config['num_classes'] if 'num_classes' in config else 10
  dset_name = config['dset_name']
  identity_init = False#config['identity_init'] if 'identity_init' in config else False
  batchsize = 128
  datasize = 10000#config['datasize']
  evalsize = datasize if not args.data_eval_size else args.data_eval_size
  steps = args.alphas

  model, loader = load_model_and_data(
    model_name, num_classes, dset_name, batchsize, datasize, True, False, False, identity_init
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

  alpha_path = os.path.join(outdir, 'alphas.npy')
  orig_losses_path = os.path.join(outdir, 'orig_losses.npy')
  rand_losses_path = os.path.join(outdir, 'rand_losses.npy')

  if os.path.isfile(alpha_path) and os.path.isfile(rand_losses_path) and os.path.isfile(orig_losses_path):
    alphas = np.load(alpha_path)
    orig_losses = np.load(orig_losses_path)
    rand_losses = np.load(rand_losses_path)
  else:
    alphas, orig_losses, rand_losses = compute_interp_data(
      model,
      loader,
      evalloader,
      init_state,
      lambda: get_model(model_name, num_classes, identity_init).cuda().state_dict(),
      final_state
    )
  mean_loss = np.mean(rand_losses, 0)
  std = np.std(rand_losses, 0)
  
  fig, ax = plt.subplots(figsize=(6,3))
  ax.set_xlim(0,1)
  ax.set_ylim(0, 4)
  ax.set_title("Interpolating initialization to final solution (CIFAR-10)", fontsize=16)
  ax.set_xlabel(r"$\alpha$", size=14)
  ax.set_ylabel("Loss", size=14)
  ax.plot(alphas, orig_losses, c='r', alpha=1)
  

  
  ax.plot(alphas, mean_loss, color='b', ls='--')
  ax.fill_between(alphas, mean_loss - std, mean_loss + std, facecolor='b', alpha=0.6)
  legend_handles=[
    Line2D([0], [0], color='b', ls='--', label='Random init'),
    Line2D([0], [0], color='r', label='Original'),
  ]
  ax.legend(handles=legend_handles, loc='lower left', fontsize=14)
  plt.tight_layout()
  if args.show:
    plt.show()
  plt.savefig(os.path.join(args.outdir, "random_init_interp.png"))

