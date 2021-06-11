import os
import itertools
import argparse
import json
import copy
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets

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

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("rundir")
parser.add_argument("outdir")
parser.add_argument("-p", "--plot_amount", type=int, default=100)
parser.add_argument("-a", "--alphas", type=int, default=50)
args = parser.parse_args()



  


if __name__ == '__main__':
  run_states = get_run_model_states(args.rundir)
  
  config = run_states['config']
  model_name = config['model_name']
  num_classes = config['num_classes'] if 'num_classes' in config else 10
  dset_name = config['dset_name']
  batchsize = 128
  datasize = config['datasize']
  steps = args.alphas

  model, loader = load_model_and_data(
    model_name, num_classes, dset_name, batchsize, datasize, True, False, False
  )
  model.cuda()
  outdir = args.outdir
  try:
    os.makedirs(outdir)
  except:
    pass
  
  init_state = run_states['init_state']
  final_state = run_states['final_state']
  plot_interp(config, run_states['metrics'], outdir)

  eval_loader = load_data(dset_name,  min(args.plot_amount, 128), args.plot_amount, True, False, False)
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
