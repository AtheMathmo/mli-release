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

from mli.models import get_activation_function, FCNet
from mli.data import load_data
from .utils import interp_networks, load_model_and_data

parser = argparse.ArgumentParser()
parser.add_argument("expdir")
parser.add_argument("outdir")
parser.add_argument("-d", "--data_eval_size", type=int, default=None)
parser.add_argument("-a", "--alphas", type=int, default=50)
parser.add_argument("--between_inits", action='store_true')
parser.add_argument("--between_opts", action='store_true')
parser.add_argument("--inits_to_opts", action='store_true')
parser.add_argument("--show", action='store_true')
args = parser.parse_args()

def get_all_run_data(expdir):
  alldirs = os.listdir(expdir)
  runs = []
  for rundir in alldirs:
    # Sacred saves all source code
    if rundir == '_sources':
      continue
    dirpath = os.path.join(expdir, rundir)
    if not os.path.isdir(dirpath):
      continue
    config_f = os.path.join(dirpath, 'config.json')
    metrics_f = os.path.join(dirpath, 'metrics.json')
    init_f = os.path.join(expdir, rundir, 'init.pt')
    final_f = os.path.join(expdir, rundir, 'final.pt')
    valid = True
    valid = valid and os.path.isfile(config_f)
    valid = valid and os.path.isfile(metrics_f)
    valid = valid and os.path.isfile(init_f)
    valid = valid and os.path.isfile(final_f)
    if not valid:
      print("Incomplete experiment output in {}".format(dirpath))
      continue
    with open(config_f, 'r') as f:
      config = json.load(f)
    runs.append({
      'config': config,
      'init': init_f,
      'final': final_f
    })
  return runs

def compute_interp_data(model, loader, evalloader, init_state_files, final_state_files):
  assert len(init_state_files) == len(final_state_files), "Number of init states must match final states"
  output = []
  for i in range(len(init_state_files)):
    init_state = torch.load(init_state_files[i])['model_state']
    final_state = torch.load(final_state_files[i])['model_state']
    alphas, metrics = interp_networks(model, init_state, final_state, loader, [evalloader], args.alphas, None, True)
    output.append({
      'loss': metrics[0]['loss'],
      'acc': metrics[0]['acc']
    })
  return alphas, output

def plot_nway_interp(alphas, data, title, figname):
  fig, ax = plt.subplots(figsize=(6,2))
  ax.set_xlim(0,1)
  ax.set_ylim(0, 2.35)
  ax.set_title(title, fontsize=16)
  ax.set_xlabel(r"$\alpha$", size=14)
  ax.set_ylabel("Loss", size=14)
  for d in data:
    ax.plot(alphas, d['loss'], c='r', alpha=0.7)
  plt.tight_layout()
  plt.savefig(os.path.join(args.outdir, "{}.pdf".format(figname)))
  plt.savefig(os.path.join(args.outdir, "{}.png".format(figname)))
  if args.show:
    plt.show()
  plt.clf()


if __name__ == '__main__':
  run_data = get_all_run_data(args.expdir)
  config = run_data[0]['config']
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
  
  if args.between_inits:
    alpha_path = os.path.join(outdir, 'inits_alphas.npy')
    data_path = os.path.join(outdir, 'inits_data.npy')
    if os.path.isfile(alpha_path) and os.path.isfile(data_path):
      alphas = np.load(alpha_path, allow_pickle=True)
      data = np.load(data_path, allow_pickle=True)
    else:
      init_state_a = []
      init_state_b = []
      for i in range(len(run_data)):
        for j in range(i + 1, len(run_data)):
          init_state_a.append(run_data[i]['init'])
          init_state_b.append(run_data[j]['init'])
      alphas, data = compute_interp_data(
        model, loader, evalloader, init_state_a, init_state_b
      )
      np.save(alpha_path, alphas)
      np.save(data_path, data)
    plot_nway_interp(alphas, data, r"Interpolation from init$\rightarrow$init", "init_nway_interp")
  if args.between_opts:
    alpha_path = os.path.join(outdir, 'opts_alphas')
    data_path = os.path.join(outdir, 'opts_data')
    if os.path.isfile(alpha_path) and os.path.isfile(data_path):
      alphas = np.load(alpha_path, allow_pickle=True)
      data = np.load(data_path, allow_pickle=True)
    else:
      final_state_a = []
      final_state_b = []
      for i in range(len(run_data)):
        for j in range(i + 1, len(run_data)):
          final_state_a.append(run_data[i]['final'])
          final_state_b.append(run_data[j]['final'])
      alphas, data = compute_interp_data(
        model, loader, evalloader, final_state_a, final_state_b
      )
      np.save(alpha_path, alphas)
      np.save(data_path, data)
    plot_nway_interp(alphas, data, r"Interpolation from optima$\rightarrow$optima", "opt_nway_interp")
  if args.inits_to_opts:
    alpha_path = os.path.join(outdir, 'inits_opts_alphas.npy')
    data_path = os.path.join(outdir, 'inits_opts_data.npy')
    if os.path.isfile(alpha_path) and os.path.isfile(data_path):
      alphas = np.load(alpha_path, allow_pickle=True)
      data = np.load(data_path, allow_pickle=True)
    else:
      init_state_a = []
      final_state_b = []
      for i in range(len(run_data)):
        for j in range(len(run_data)):
          init_state_a.append(run_data[i]['init'])
          final_state_b.append(run_data[j]['final'])
      alphas, data = compute_interp_data(
        model, loader, evalloader, init_state_a, final_state_b
      )
      np.save(alpha_path, alphas)
      np.save(data_path, data)
    plot_nway_interp(alphas, data, r"Interpolation from init$\rightarrow$optima", "init_opt_nway_interp")
