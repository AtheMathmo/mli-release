import argparse

import matplotlib.pyplot as plt
plt.style.use('seaborn')

from mli.models import FCNet, interpolate_state, get_activation_function
from mli.data import load_data
from mli.metrics.gauss_len import compute_avg_gauss_length, compute_avg_gauss_length_bn

from .utils import *

parser = argparse.ArgumentParser()
parser.add_argument("rundir")
parser.add_argument("outdir")
parser.add_argument("-d", "--data_eval_size", type=int, default=None)
parser.add_argument("-a", "--alphas", type=int, default=50)
parser.add_argument("--compute_gl", action="store_true")
args = parser.parse_args()


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

  if args.compute_gl:
    alphas = np.linspace(0, 1, steps, endpoint=True)
    if not model.use_batchnorm:
      # This version is quicker
      avg_gl = compute_avg_gauss_length(model, init_state, final_state, np.linspace(0, 1, steps, endpoint=True), evalloader)
    else:
      # Slower but handles batch norm correctly
      avg_gl = compute_avg_gauss_length_bn(model, init_state, final_state, np.linspace(0, 1, steps, endpoint=True), loader, evalloader, bn_warm_steps=1)
    print("Avg Gauss Length: {}".format(avg_gl))
    with open(os.path.join(outdir, "gl.txt"), 'w') as f:
      f.write(str(avg_gl))
