import os
import itertools
import argparse
import functools
import json

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


import tqdm

import mli.models as models
from mli.models import interpolate_state, warm_bn, FCNet, get_activation_function
from mli.data import load_data
from mli.metrics import param_dist

parser = argparse.ArgumentParser()
parser.add_argument("rundir")
parser.add_argument("outdir")
parser.add_argument("-d", "--data_eval_size", type=int, default=None)
parser.add_argument("-a", "--alphas", type=int, default=50)
parser.add_argument("--gridsteps", default=10,type=int)
args = parser.parse_args()

def get_model(hsizes, act_fn, batch_norm):
  return FCNet(784, hsizes, act_fn=act_fn, batch_norm=batch_norm)

def load_model_and_data(hsizes, dset_name, act_fn, batch_norm, batchsize, datasize, train=True, shuffle=False):
  model = get_model(hsizes, act_fn, batch_norm)
  loader = load_data(dset_name, batchsize, datasize, train, shuffle)
  return model, loader



def get_run_model_states(rundir):
  """TODO: for large models we can't load all of these in memory at once. We should just store the paths.
  """
  entries = os.listdir(rundir)

  ret = {}
  config_f = os.path.join(rundir, 'config.json')
  with open(config_f, 'r') as f:
    ret['config'] = json.load(f) 

  metric_f = os.path.join(rundir, 'metrics.json')
  with open(metric_f, 'r') as f:
    ret['metrics'] = json.load(f)

  init_f = os.path.join(rundir, 'init.pt')
  with open(metric_f, 'r') as f:
    ret['init_state'] = torch.load(init_f)['model_state']

  final_f = os.path.join(rundir, 'final.pt')
  with open(metric_f, 'r') as f:
    ret['final_state'] = torch.load(final_f)['model_state']
    
  ret['states'] = {}
  for entry in entries:
    if not entry.startswith('model_'):
      continue
    path = os.path.join(rundir, entry)
    ret['states'][entry.split('_')[1].split('.')[0]] = torch.load(path)['model_state']

  return ret


def eval_model(model, loader, cuda):
  model.eval()
  loss = 0.0
  acc = 0.0
  with torch.no_grad():
    for x,y in loader:
      if cuda:
        x,y = x.cuda(), y.cuda()
      logits = model(x)
      preds = logits.argmax(1)
      acc += (preds == y).float().sum().item()
      b_loss = F.cross_entropy(logits, y)
      loss += b_loss.item() * x.shape[0]
  model.train()
  return loss / len(loader.dataset), acc / len(loader.dataset)


def interp_networks(model, init_state, final_state, train_loader, eval_loader, alpha_steps, cuda):
  if cuda:
    model.cuda()

  losses = []
  accs = []
  alpha_range = np.linspace(0, 1, alpha_steps, endpoint=True)
  if model.use_batchnorm:
    print('Model uses batchnorm, will take a while...')
  for alpha in tqdm.tqdm(alpha_range):
    interpolate_state(model.state_dict(), init_state, final_state, alpha)
    if model.use_batchnorm:
      warm_bn(model, train_loader, cuda)
    loss, acc = eval_model(model, eval_loader, cuda)
    losses.append(loss)
    accs.append(acc)
  return alpha_range, np.array(losses), np.array(accs)


def compute_ortho_2d_basis(state1, state2, state3):
  dx = 0
  u = {}
  for layer in state1:
    ab = state3[layer] - state1[layer]
    if 'running' not in layer: # Skip batch norm running stats
      dx += (ab * ab).sum()
    u[layer] = ab
  dx = torch.sqrt(dx)
  for layer in u:
    u[layer] = u[layer] / dx

  dy = 0
  v = {}
  proj = 0
  for layer in state1:
    bc = state2[layer] - state1[layer]
    proj += (bc * u[layer]).sum()
  for layer in state1:
    bc = state2[layer] - state1[layer]
    bc = bc - proj * u[layer]
    if 'running' not in layer: # Skip batch norm running stats
      dy += (bc * bc).sum()
    v[layer] = bc
  dy = torch.sqrt(dy)
  for layer in v:
    v[layer] = v[layer] / dy
  return u, v, dx, dy

def compute_grid_state(init_state, x_step, u, y_step, v):
  grid_state = {}
  for layer in init_state:
    grid_state[layer] = init_state[layer] + x_step * u[layer] + y_step * v[layer]
  return grid_state


def project_point(point, origin, u, v):
  '''
  Projects a given point onto a 2-d plane spanned by given basis vectors
  '''
  projection = torch.tensor([0.0, 0.0])
  for layer in point:
    if 'running' not in layer: # Skip batch norm running stats
      projection += torch.tensor([((point[layer] - origin[layer]) * u[layer]).sum(), ((point[layer] - origin[layer]) * v[layer]).sum()])
  return projection

def get_grid_steps(margins, total=10):
  '''
  Margins should be (left, right, bottom, top) tuple
  '''
  alphas = np.linspace(0.0 - margins[0], 1.0 + margins[1], total, dtype=np.float32)
  betas = np.linspace(0.0 - margins[2], 1.0 + margins[3], total, dtype=np.float32)
  return alphas, betas

def compute_grid_loss(
    model, loader, evalloader, gridsteps, dx, dy, u, v, init_state,
    force_recompute=False
  ):
  xgrid_path = os.path.join(outdir, 'x_grid.npy')
  ygrid_path = os.path.join(outdir, 'y_grid.npy')
  gridacc_path = os.path.join(outdir, 'grid_acc.npy')
  gridloss_path = os.path.join(outdir, 'grid_loss.npy')

  if not os.path.isfile(gridloss_path) or force_recompute:
    print('Evaluating model on grid...')
    alphas, betas = get_grid_steps((0.3,0.3,0.3,0.3), gridsteps)
    x_grid = np.zeros(gridsteps)
    y_grid = np.zeros(gridsteps)
    grid_loss = np.zeros((gridsteps, gridsteps))
    grid_acc = np.zeros((gridsteps, gridsteps))
    if model.use_batchnorm:
      print("WARNING: Model uses batchnorm. This will take a very long time.")
    for i, alpha in enumerate(alphas):
      x_step = alpha * dx
      x_grid[i] = x_step
      for j, beta in enumerate(betas):
        print("Evaluating {}/{}".format(i * len(betas) + j, len(alphas) * len(betas)))
        y_step = beta * dy
        y_grid[j] = y_step
        p = compute_grid_state(init_state, x_step, u, y_step, v)
        model.load_state_dict(p)
        if model.use_batchnorm:
          warm_bn(model, loader, True)
        loss, acc = eval_model(model, evalloader, True)
        grid_acc[i,j] = acc
        grid_loss[i,j] = loss
    np.save(xgrid_path, x_grid)
    np.save(ygrid_path, y_grid)
    np.save(gridacc_path, grid_acc)
    np.save(gridloss_path, grid_loss)
  else:
    x_grid = np.load(xgrid_path)
    y_grid = np.load(ygrid_path)
    grid_acc = np.load(gridacc_path)
    grid_loss = np.load(gridloss_path)
  return x_grid, y_grid, grid_acc, grid_loss

def plot_interp(ax, alphas, losses, **kwargs):
  ax.plot(alphas, losses, lw=4, **kwargs)

def plot_loss(fig, ax, model, loader, evalloader, runstates, gridsteps, outdir, figpath):
  """Steps:

  1. Find the 2D basis
  2. Produce a grid over the trajectory
  3. Evaluate loss at each point on grid
  4. Highlight the init, middle, and final param position
  """
  

  init_state = runstates['init_state']
  init2_state = runstates['init2']
  final_state = runstates['final_state']
  with torch.no_grad():
    print('Computing projections')
    u,v,dx,dy = compute_ortho_2d_basis(init_state, init2_state, final_state)

    print('Projecting trajectory')
    init_project = project_point(init_state, init_state, u, v)
    final_project = project_point(final_state, init_state, u, v)
    init2_project = project_point(init2_state, init_state, u, v)
    # Plot the points defining the plane
    ax.plot([init_project[0]], [init_project[1]], marker='o', markersize=15, label='Init 1', c='crimson')
    ax.plot([init2_project[0]], [init2_project[1]], marker='o', markersize=15, label='Init 2', c='navy')
    ax.plot([final_project[0]], [final_project[1]], marker='x', markersize=25, mew=2, label='Final 1', c='crimson')


    # Plot the training trajectory
    train_traj = [init_project.cpu().numpy()]
    stateidxs = [int(s) for s in run_states['states'].keys()]
    stateidxs.sort()
    for k in stateidxs:
      state = run_states['states'][str(k)]
      train_traj.append(project_point(state, init_state, u, v).cpu().numpy())
    train_traj = np.array(train_traj)
    ax.plot(train_traj[:,0], train_traj[:,1], marker='.', lw=4, ls='--', c='orange', label='Optimization')

    # # Plot interpolation paths
    ax.plot([init_project[0], final_project[0]], [init_project[1], final_project[1]], c='crimson', lw=4, alpha=0.7)
    ax.plot([init2_project[0], final_project[0]], [init2_project[1], final_project[1]], c='navy', lw=4, alpha=0.7)
    
    x_grid, y_grid, grid_acc, grid_loss = compute_grid_loss(
      model, loader, evalloader,
      gridsteps, dx, dy, u, v,
      init_state
    )
  cmap = plt.cm.Greens_r
  X,Y = np.meshgrid(x_grid, y_grid)
  levels = np.linspace(0, 3, 20)
  CS = ax.contourf(X, Y, grid_loss.T, cmap=cmap, levels=levels, vmin=0, vmax=3, extend='max')
  cbar = fig.colorbar(CS, ax=ax, ticks=np.linspace(0.0, 3.0, 7))


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
  
  # Get data for second initialization
  ralpha_path = os.path.join(args.outdir, 'randinit_alphas.npy')
  rloss_path = os.path.join(args.outdir, 'randinit_losses.npy')
  racc_path = os.path.join(args.outdir, 'randinit_accs.npy')
  rinit_path = os.path.join(args.outdir, 'randinit2.pt')
  if not os.path.isfile(ralpha_path):
    import pdb; pdb.set_trace()
    rand_init2 = get_model(hsizes, act_fn, use_batchnorm).cuda().state_dict()
    alphas, losses, accs = interp_networks(model, rand_init2, final_state, loader, evalloader, args.alphas, True)
    np.save(ralpha_path, alphas)
    np.save(rloss_path, losses)
    np.save(racc_path, accs)
    torch.save(rand_init2, rinit_path)
  else:
    alphas = np.load(ralpha_path)
    losses = np.load(rloss_path)
    accs = np.load(racc_path)
    rand_init2 = torch.load(rinit_path)

  run_states['init2'] = rand_init2
  ## Compute global dy max
  # print('Computing global basis scaling')
  # max_dy = 0.0
  # for k in run_states['states']:
  #   state = run_states['states'][k]
  #   u,v,dx,dy = compute_ortho_2d_basis(init_state, state, final_state)
  #   if dy > max_dy:
  #     max_dy = dy
  # print('Done!')
  fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
  plot_loss(fig, axs[0], model, loader, evalloader, run_states, args.gridsteps, outdir, 'landscape.png')
  axs[1].set_xlim(0, 1)
  axs[1].set_ylim(0, 3.5)

  final_dist = param_dist(init_state, final_state)
  dists = [0]
  stateidxs = [int(s) for s in run_states['states'].keys()]
  stateidxs.sort()
  for k in stateidxs:
    state = run_states['states'][str(k)]
    dist_to_final = param_dist(state, final_state)
    dists.append(1.0 - dist_to_final / final_dist)
  # train_alphas = np.array(run_states['metrics']['train.wdist']['values']) / final_dist
  # train_alphas = np.insert(train_alphas, 0, [0])
  train_losses = np.array(run_states['metrics']['train.loss']['values'])
  train_losses = np.insert(train_losses, 0, run_states['metrics']['interpolation.loss']['values'][0])
  # plot_interp(axs[1], dists, train_losses, label='Training', c='orange', marker='o', alpha=0.7)
  plot_interp(
    axs[1],
    run_states['metrics']['interpolation.alpha']['values'],
    run_states['metrics']['interpolation.loss']['values'],
    label=r'Init 1 $\rightarrow$ Opt 1',
    c='crimson'
  )
  plot_interp(axs[1], alphas, losses, label=r'Init 2 $\rightarrow$ Opt 1', c='navy')
  

  axs[1].legend(loc='upper right', fontsize=20, frameon=True, fancybox=True, framealpha=0.5)
  legend_handles = [
    Line2D([0], [0], marker='.', ls='--', c='orange', label='Optimizer\npath'),
    Line2D([0], [0], marker='o', markerfacecolor='crimson', color=(0,0,0,0), markersize=15, label='Init 1'),
    Line2D([0], [0], marker='o', markerfacecolor='navy', color=(0,0,0,0), markersize=15, label='Init 2'),
    Line2D([0], [0], marker='x', markerfacecolor='crimson', markeredgecolor='crimson', color=(0,0,0,0), markersize=15, mew=2, label='Opt 1')
  ]
  axs[0].legend(handles=legend_handles, fontsize=20, loc='upper right', frameon=True, fancybox=True, framealpha=0.5)

  axs[0].set_title("Loss Landscape", fontsize=26)
  axs[1].set_title("Loss Interpolation", fontsize=26)
  axs[0].set_xlabel('Weight direction 1', fontsize=20)
  axs[0].set_ylabel('Weight direction 2', fontsize=20)
  axs[1].set_xlabel(r'$\alpha$', fontsize=20)
  axs[1].set_ylabel('Training loss', fontsize=20)
  plt.tight_layout()
  plt.savefig(os.path.join(outdir, 'loss_landscape_compare.pdf'))

