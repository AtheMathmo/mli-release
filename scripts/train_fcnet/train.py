import torch
import os
import tqdm
import copy

import torch.nn.functional as F
import numpy as np

from munch import Munch

from sacred import Experiment
from mli.data import load_data, corrupt_dataset_labels
from mli.optim import get_optimizer
from mli.models import get_activation_function, get_loss_fn, interpolate_state
from mli.models import FCNet, warm_bn
from mli.metrics.gauss_len import compute_avg_gauss_length, compute_avg_gauss_length_bn
from mli.sacred import SlurmFileStorageObserver
from mli.metrics import param_dist

from mli_eval.model.interp import interp_networks
from mli_eval.model.loss import EvalClassifierLoss

EXPERIMENT_NAME = "mli_fcnet"
RUN_DIR = "runs"
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(SlurmFileStorageObserver(os.path.join(RUN_DIR, EXPERIMENT_NAME)))


@ex.config
def get_config():
    # Data Config
    dset_name = "fashionmnist"
    datasize = 60000
    random_label_proportion = 0.0
    batchsize = 512
    corrupt_on_epoch = -1

    # Model Config
    hsizes = [100, 10]
    act_fn = "relu"
    loss_fn = "ce"
    regularization = None
    regscale = 1
    outR_reg = 1
    use_batchnorm = False

    # Initialization
    init_type = "default"

    # Optimizer Config
    epochs = 10
    optim_name = "sgd"
    lr = 0.1
    beta = 0.9

    # Misc
    alpha_steps = 50
    cuda = True
    min_loss_threshold = 1
    min_loss_epoch_check = 10
    log_wdist = True

    # Experiment Config
    tag = "fcnet"
    seed = 0
    save_freq = 25
    eval_gl = True


@ex.capture
def get_run_id(_run):
    return _run._id


@ex.capture
def load_data_captured(dset_name, batchsize, datasize, train=True):
    return load_data(dset_name, batchsize, datasize, train=train)


@ex.capture
def get_optimizer_captured(optim_name, lr, beta):
    return get_optimizer(optim_name, lr, beta)


@ex.capture
def get_model(hsizes, act_fn, init_type, use_batchnorm):
    model = FCNet(784, hsizes, act_fn=get_activation_function(act_fn), init_type=init_type, batch_norm=use_batchnorm)
    return model


@ex.capture
def compute_loss(model, out, targets, regularization, loss_fn, regscale, outR_reg):
    loss = get_loss_fn(loss_fn)(out, targets)
    if regularization is None:
        return loss
    elif regularization == "l1":
        l1_reg = 0.0
        scale = 1.0
        for param in model.parameters():
            l1_reg += scale * F.l1_loss(param, torch.zeros_like(param))
        return loss + l1_reg
    elif regularization == "outR":
        reg = 0.0
        for param in model.parameters():
            reg += regscale * (torch.linalg.norm(param) - outR_reg) ** 2
        return loss + reg


def eval_loss(model, loader, cuda):
    model.eval()
    loss = 0.0
    acc = 0.0
    with torch.no_grad():
        for x, y in loader:
            if cuda:
                x, y = x.view(-1, 784).cuda(), y.cuda()
            logits = model(x)
            preds = logits.argmax(1)
            acc += (preds == y).float().sum().item()
            b_loss = F.cross_entropy(logits, y)
            loss += b_loss.item() * x.shape[0]
    model.train()
    return loss / len(loader.dataset), acc / len(loader.dataset)


def train_step(model, optimizer, x, y, compute_acc=True):
    optimizer.zero_grad()
    logits = model(x)
    loss = compute_loss(model, logits, y)
    loss.backward()
    optimizer.step()

    ret = {
        "loss": loss.item(),
    }

    if compute_acc:
        preds = logits.argmax(1)
        acc = (preds == y).float().mean()
        ret["acc"] = acc.item()
    return ret


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
        for x, y in pbar:
            if cfg.cuda:
                x, y = x.view(-1, 784).cuda(), y.cuda()
            if cfg.loss_fn == "ce":
                train_metrics.append(train_step(model, optimizer, x, y, True))
                mean_loss = np.mean([m["loss"] for m in train_metrics])
                pbar.set_description(
                    "Epoch {:d} train | loss = {:0.6f}, acc = {:0.4f}".format(
                        epoch,
                        mean_loss,
                        np.mean([m["acc"] for m in train_metrics]),
                    )
                )
            elif cfg.loss_fn == "recon":
                train_metrics.append(train_step(model, optimizer, x, x, False))
                mean_loss = np.mean([m["loss"] for m in train_metrics])
                pbar.set_description(
                    "Epoch {:d} train | loss = {:0.6f}".format(
                        epoch,
                        mean_loss,
                    )
                )
            else:
                raise Exception("Invalid loss function given")

        if cfg.loss_fn == "ce":
            _run.log_scalar("train.loss", np.mean([m["loss"] for m in train_metrics]))
            _run.log_scalar("train.acc", np.mean([m["acc"] for m in train_metrics]))
        elif cfg.loss_fn == "recon":
            _run.log_scalar("train.loss", np.mean([m["loss"] for m in train_metrics]))
        else:
            raise Exception("Invalid loss function given")

        if cfg.log_wdist:
            _run.log_scalar("train.norm_wdist", param_dist(model.state_dict(), init_state, True))
            _run.log_scalar("train.wdist", param_dist(model.state_dict(), init_state, False))

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

    # Save the final state
    final_outfile = os.path.join(checkpoint_dir, "final.pt")
    torch.save({
        "model_state": final_state,
    }, final_outfile)
    return init_state, final_state, losses


def get_run_dir():
    rundir = os.getenv("SLURM_JOB_ID")
    if rundir is None:
        rundir = os.getenv("SLURM_ARRAY_JOB_ID")
    if rundir is None:
        rundir = get_run_id()
    return rundir


@ex.automain
def main(seed, _run):
    cfg = Munch.fromDict(_run.config)

    train_loader = load_data_captured()
    eval_loader = load_data_captured(train=False)
    model = get_model()
    if cfg.cuda:
        model = model.cuda()
    try:
        optimizer = get_optimizer_captured()(model.parameters())
    except AttributeError:
        optimizer = get_optimizer_captured()(model)

    # Train network
    init_state, final_state, _ = train_network(
        model, train_loader, optimizer, cfg, _run
    )
    # Evaluate interpolation path of networks
    alphas, metrics = interp_networks(
        model, init_state, final_state, 
        train_loader, [train_loader, eval_loader],
        cfg.alpha_steps, EvalClassifierLoss(), cfg.cuda
    )
    for i in range(len(metrics[0]['loss'])):
        _run.log_scalar("train.interpolation.loss",  metrics[0]['loss'][i])
        _run.log_scalar("train.interpolation.acc", metrics[0]['acc'][i])
        _run.log_scalar("train.interpolation.alpha", alphas[i])

        _run.log_scalar("eval.interpolation.loss", metrics[1]['loss'][i])
        _run.log_scalar("eval.interpolation.acc", metrics[1]['acc'][i])
        _run.log_scalar("eval.interpolation.alpha", alphas[i])
    
    # Evaluate the gauss length of the interpolation path
    if cfg.eval_gl:
        if not model.use_batchnorm:
            # This version is quicker
            avg_gl = compute_avg_gauss_length(model, init_state, final_state, alphas, eval_loader)
        else:
            # Slower but handles batch norm correctly
            avg_gl = compute_avg_gauss_length_bn(model, init_state, final_state, alphas, train_loader, eval_loader,
                                                bn_warm_steps=1)
        _run.log_scalar("gauss_len", avg_gl)
