import torch
import torch.nn.functional as F

import os
import tqdm
import copy
from munch import Munch

import numpy as np

from sacred import Experiment


from mli.data import load_data, corrupt_dataset_labels
from mli.optim import get_optimizer
from mli.models import get_activation_function, get_loss_fn, interpolate_state
from mli.models import FCNet
from mli.sacred import SlurmFileStorageObserver

from mli_eval.model.interp import interp_networks
from mli_eval.model.loss import EvalAutoencoderLoss


EXPERIMENT_NAME = "mli_ae_fc"
RUN_DIR = "runs"
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(SlurmFileStorageObserver(os.path.join(RUN_DIR, EXPERIMENT_NAME)))


@ex.config
def get_config():
  ## Data Config
  dset_name = "mnist"
  datasize = 60000
  random_label_proportion = 0.0
  batchsize = 512
  corrupt_on_epoch = 0

  ## Model Config
  hsizes = [512,10,512,784]
  act_fn = "relu"
  loss_fn = "recon"
  regularization = None

  # Initialization
  init_type = "default"

  ## Optimizer Config
  epochs = 200
  optim_name = "sgd"
  lr = 0.1
  beta = 0.9

  ## Misc
  alpha_steps = 50
  cuda = True
  min_loss_threshold = 55
  min_loss_epoch_check = 10

  ## Experiment Config
  tag = "ae_fc"
  seed = 17
  save_freq = 25


@ex.capture
def get_run_id(_run):
    return _run._id


@ex.capture
def load_data_captured(dset_name, batchsize, datasize, train=True):
    return load_data(dset_name, batchsize, datasize, train)


@ex.capture
def get_optimizer_captured(optim_name, lr, beta):
    return get_optimizer(optim_name, lr, beta)


@ex.capture
def get_model(hsizes, act_fn, init_type):
    model = FCNet(784, hsizes, act_fn=get_activation_function(act_fn), init_type=init_type, batch_norm=False)
    return model


@ex.capture
def compute_loss(model, out, targets, regularization, loss_fn):
    loss = get_loss_fn(loss_fn)(out, targets)
    if regularization is None:
        return loss
    elif regularization == "l1":
        l1_reg = 0.0
        scale = 1.0
        for param in model.parameters():
            l1_reg += scale * F.l1_loss(param, torch.zeros_like(param))
        return loss + l1_reg


def eval_loss(model, loader, cuda):
    model.eval()
    loss = 0.0
    with torch.no_grad():
      for x,_ in loader:
          if cuda:
              x = x.view(-1, 784).cuda()
          logits = model(x)
          loss += compute_loss(model, logits, x).item()
    model.train()
    return loss / len(loader.dataset)


def train_step(model, optimizer, x):
    optimizer.zero_grad()
    logits = model(x)
    loss = compute_loss(model, logits, x) / x.shape[0]
    loss.backward()
    optimizer.step()
    return {
      "loss": loss.item(),
    }

def train_network(model, loader, optimizer, cfg, _run):
    init_state = copy.deepcopy(model.state_dict())
    checkpoint_dir = os.path.join(RUN_DIR, EXPERIMENT_NAME, get_run_dir())

    start_epoch = 0
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    if os.path.isfile(checkpoint_path):
        print("Found an existing checkpoint. Loading state...")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("Training from epoch {}".format(start_epoch))
    else:
        # First time on this run
        # Save the initial state
        print("No checkpoint found. Training from scratch.")
        init_outfile = os.path.join(checkpoint_dir, "init.pt")
        torch.save({
          "model_state": init_state,
        }, init_outfile)

    losses = []
    for epoch in range(start_epoch, cfg.epochs):
        if cfg.corrupt_on_epoch == epoch:
            corrupt_dataset_labels(loader, cfg.random_label_proportion)
            targetfile = os.path.join(RUN_DIR, EXPERIMENT_NAME, get_run_dir(), "targets")
            np.save(targetfile, loader.dataset.targets.numpy())

        model.train()
        train_metrics = []
        pbar = tqdm.tqdm(loader)
        mean_loss = 0
        for x,_ in pbar:
            if cfg.cuda:
                x = x.view(-1,784).cuda()
            train_metrics.append(train_step(model, optimizer, x))
            mean_loss = np.mean([m["loss"] for m in train_metrics])
            pbar.set_description(
                "Epoch {:d} train | loss = {:0.6f}".format(
                epoch,
                mean_loss,
                )
            )
        if cfg.loss_fn == "recon":
            _run.log_scalar("train.loss", np.mean([m["loss"] for m in train_metrics]))
        else:
            raise Exception("Invalid loss function given")
        if epoch > cfg.min_loss_epoch_check and mean_loss > cfg.min_loss_threshold:
            print("Loss threshold not reached by epoch %s" % cfg.min_loss_epoch_check)
            print("Breaking out of training early...")
            break
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, checkpoint_path)
        if cfg.save_freq > 0 and epoch % cfg.save_freq == 0:
            outfile = os.path.join(checkpoint_dir, "model_{}.pt".format(epoch))
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict()
            }, outfile)

    final_state = copy.deepcopy(model.state_dict())

    ## Save the final state
    final_outfile = os.path.join(checkpoint_dir, "final.pt")
    torch.save({
      "model_state": final_state,
    }, final_outfile)
    return init_state, final_state, losses

def eval_interp(model, init_state, final_state, train_loader, alpha_steps, cuda, _run):
    eval_loader = load_data_captured(train=False)
    alphas, metrics = interp_networks(
          model, init_state, final_state,
          train_loader, [train_loader, eval_loader],
          alpha_steps, EvalAutoencoderLoss(), cuda)
    train_losses = metrics[0]["loss"]
    eval_losses = metrics[1]["loss"]
    for i in range(len(train_losses)):
        _run.log_scalar("train.interpolation.loss", train_losses[i])
        _run.log_scalar("train.interpolation.alpha", alphas[i])

        _run.log_scalar("eval.interpolation.loss", eval_losses[i])
        _run.log_scalar("eval.interpolation.alpha", alphas[i])


def get_run_dir():
    rundir = os.getenv("SLURM_JOB_ID")
    if rundir is None:
      rundir = os.getenv("SLURM_ARRAY_JOB_ID")
    if rundir is None:
      rundir = get_run_id()
    return rundir


@ex.automain
def main(_run):
    cfg = Munch.fromDict(_run.config)
    train_loader = load_data_captured()
    model = get_model()
    if cfg.cuda:
        model = model.cuda()
    optimizer = get_optimizer_captured()(model.parameters())

    init_state, final_state, _ = train_network(
      model, train_loader, optimizer, cfg, _run
    )
    eval_interp(
      model, init_state, final_state,
      train_loader, cfg.alpha_steps, cfg.cuda, _run
    )
