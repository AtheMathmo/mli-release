import matplotlib.pyplot as plt
plt.style.use('seaborn')

import torch
import torch.nn.functional as F
import numpy as np
import tqdm

from mli.models import FCNet, interpolate_state, get_activation_function, warm_bn
from mli.data import load_data
import mli.metrics as metrics_utils

from mli_eval.processing.experiments import get_run_stats, get_run_model_states
from mli_eval.model.eval import eval_model, eval_model_per_example
from mli_eval.model.interp import interp_networks, interp_networks_eval_examples, interpolate_state

def load_model_and_data(hsizes, dset_name, act_fn, batch_norm, batchsize, datasize, train=True, shuffle=False):
    model = FCNet(784, hsizes, act_fn=act_fn, batch_norm=batch_norm)
    loader = load_data(dset_name, batchsize, datasize, train, shuffle)
    return model, loader
