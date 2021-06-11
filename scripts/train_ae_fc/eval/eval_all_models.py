import os
import itertools
import argparse

import json
import tqdm

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets

from mli.models import FCNet, interpolate_state, get_activation_function
from mli.data import load_data
from mli.metrics.gauss_len import compute_avg_gauss_length, compute_avg_gauss_length_bn
import mli.metrics as metrics_utils

from eval.utils import *
from .utils import get_run_stats, interp_networks_eval_examples, load_model_and_data
from mli_eval.plotting.interp import plot_interp
from mli_eval.processing.experiments import get_monotonicity_metrics, summarize_metrics

parser = argparse.ArgumentParser()
parser.add_argument("expdir")
parser.add_argument("outdir")
parser.add_argument("-d", "--data_eval_size", type=int, default=None)
parser.add_argument("-a", "--alphas", type=int, default=50)
parser.add_argument("--compute_gl", action="store_true")
args = parser.parse_args()


def process_experiments(expdir, outdir):
    try:
        os.makedirs(outdir)
    except:
        pass
    all_configs, all_metrics = get_run_stats(expdir)
    for i in range(len(all_configs)):
        config = all_configs[i]
        hsizes = config["hsizes"]
        dset_name = config["dset_name"]
        act_fn = get_activation_function(config["act_fn"])
        use_batchnorm = config["use_batchnorm"] if "use_batchnorm" in config else False
        datasize = config["datasize"]
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

        init_state = torch.load(os.path.join(expdir, config["run_id"], "init.pt"))["model_state"]
        final_state = torch.load(os.path.join(expdir, config["run_id"], "final.pt"))["model_state"]

        if args.compute_gl:
            alphas = np.linspace(0, 1, steps, endpoint=True)
            if not model.use_batchnorm:
                # This version is quicker
                avg_gl = compute_avg_gauss_length(model, init_state, final_state, np.linspace(0, 1, steps, endpoint=True), evalloader)
            else:
                # Slower but handles batch norm correctly
                avg_gl = compute_avg_gauss_length_bn(model, init_state, final_state, np.linspace(0, 1, steps, endpoint=True), loader, evalloader, bn_warm_steps=1)
            all_metrics[i]["gauss_length"] = avg_gl
  
    summary = summarize_metrics(
      all_configs, all_metrics, metric_summaries=[(np.min, "train.loss")]
    )
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f)

    monotonicity_summary = get_monotonicity_metrics(all_configs, all_metrics)
    with open(os.path.join(outdir, "monotonicity_metrics.json"), "w") as f:
        json.dump(monotonicity_summary, f)


if __name__ == "__main__":
    process_experiments(args.expdir, args.outdir)
