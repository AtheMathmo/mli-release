import os
import json
import functools

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import torch
import torch.nn.functional as F
import numpy as np
import tqdm

from mli.models import interpolate_state, get_activation_function
import mli.models as models
from mli.data import load_data
import mli.metrics as metrics_utils
from mli.models import warm_bn

from mli_eval.processing.experiments import get_run_stats, get_run_model_states
from mli_eval.model.eval import eval_model, eval_model_per_example
from mli_eval.model.interp import interp_networks, interp_networks_eval_examples, interpolate_state

MODEL_MAP = {
  'resnet-20': models.resnet20,
  'fixup_resnet-20': models.fixup_resnet20,
  'resnet-20-nobn': functools.partial(models.resnet20, use_batchnorm=False),
  'resnet-32': models.resnet32,
  'fixup_resnet-32': models.fixup_resnet32,
  'resnet-32-nobn': functools.partial(models.resnet32, use_batchnorm=False),
  'resnet-44': models.resnet44,
  'fixup_resnet-44': models.fixup_resnet44,
  'resnet-44-nobn': functools.partial(models.resnet44, use_batchnorm=False),
  'resnet-56': models.resnet56,
  'fixup_resnet-56': models.fixup_resnet56,
  'resnet-56-nobn': functools.partial(models.resnet56, use_batchnorm=False),
  'resnet-110': models.resnet110,
  'fixup_resnet-110': models.fixup_resnet110,
  'resnet-110-nobn': functools.partial(models.resnet110, use_batchnorm=False),
}

def get_model(model_name, num_classes, identity_init=False):
  if 'fixup' not in model_name:
    return MODEL_MAP[model_name](num_classes=num_classes, identity_init=identity_init)
  else:
    return MODEL_MAP[model_name](num_classes=num_classes)

def load_model_and_data(model_name, num_classes, dset_name, batchsize, datasize, train=True, shuffle=False, random_augment_train=True, identity_init=False):
  model = get_model(model_name, num_classes, identity_init=identity_init)
  loader = load_data(dset_name, batchsize, datasize, train, shuffle, random_augment_train)
  return model, loader

