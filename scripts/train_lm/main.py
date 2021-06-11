import mli.data.corpus as corp
import time
import math
import os
import torch.nn as nn
import torch.onnx
import copy

from mli.metrics.gauss_len import *
from sacred import Experiment
from mli.sacred import SlurmFileStorageObserver
from mli.models import LILSTM
from mli.models import interpolate_state
from mli.models import LITransformerModel
from mli.metrics.dist import *


EXPERIMENT_NAME = "mli_lm"
RUN_DIR = "runs"
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(SlurmFileStorageObserver(os.path.join(RUN_DIR, EXPERIMENT_NAME)))


@ex.config
def get_config():
    # Data Config
    data = "data/"

    # Model Config
    model = "lstm"
    nhead = 2
    emsize = 200
    nhid = 200
    nlayers = 2
    batch_size = 20
    bptt = 35
    dropout = 0

    # Initialization
    init_type = "default"

    # Optimizer Config
    optimizer = "sgd"
    epochs = 5
    lr = 10
    clip = 0.25

    # Misc
    alpha_steps = 5
    cuda = True
    log_wdist = True

    log_interval = 500

    # Experiment Config
    tag = "lm"
    seed = 0
    cd = 200


@ex.capture
def get_run_id(_run):
    return _run._id


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn"t cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(model, eval_batch_size, corpus, bptt, criterion, data_source):
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if not isinstance(model, LITransformerModel):
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            if isinstance(model, LITransformerModel):
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
                output = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train(model, epoch, corpus, clip, lr, train_data, bptt, criterion, log_interval, batch_size, optimizer, _run):
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if not isinstance(model, LITransformerModel):
        hidden = model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        model.zero_grad()
        if isinstance(model, LITransformerModel):
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            output = output.view(-1, ntokens)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        # for p in model.parameters():
        #     p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print("| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | "
                    "loss {:5.2f} | ppl {:8.2f}".format(
                epoch, batch, len(train_data) // bptt, lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def get_model(model, ntokens, emsize, nhead, nhid, nlayers, dropout, device):
    if model == "transformer":
        model = LITransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    else:
        model = LILSTM(ntokens, emsize, nhid, nlayers).to(device)
    return model


def interp_networks(model, ntokens, emsize, nhead, nhid, nlayers, dropout, device, train_batch_size,
                    eval_batch_size, corpus, bptt, criterion, train_data, val_data, init_state,
                    final_state, alpha_steps, cuda, _run):
    model = get_model(model, ntokens, emsize, nhead, nhid, nlayers, dropout, device)
    if cuda:
        model.cuda()

    alpha_range = np.linspace(0, 1, alpha_steps, endpoint=True)
    for alpha in tqdm.tqdm(alpha_range):
        interpolate_state(model.state_dict(), init_state, final_state, alpha)
        train_loss = evaluate(model, train_batch_size, corpus, bptt, criterion, train_data)
        _run.log_scalar("train.interpolation.loss", train_loss)
        _run.log_scalar("train.interpolation.alpha", alpha)

        val_loss = evaluate(model, eval_batch_size, corpus, bptt, criterion, val_data)
        _run.log_scalar("val.interpolation.loss", val_loss)
        _run.log_scalar("val.interpolation.alpha", alpha)


def compute_avg_gauss_length(model, ntokens, batch_size, init_state, final_state,
                             alphas, bptt, data_source):
    gauss_length_sum = 0.0
    data_count = 0.0
    model.eval()

    if not isinstance(model, LITransformerModel):
        hidden = model.init_hidden(batch_size)

    for i in range(0, data_source.size(0) - 1, bptt):
        data, targets = get_batch(data_source, i, bptt)
        data_count += data.shape[0]
        if isinstance(model, LITransformerModel):
            gl = gauss_length_func(model, ntokens, None, init_state, final_state, data, True)
        else:
            gl = gauss_length_func(model, hidden, init_state, final_state, data, True)
        gauss_lengths = trapez_integrate(gl, alphas)
        gauss_length_sum += gauss_lengths.sum()
    avg_gauss_length = gauss_length_sum / data_count
    return avg_gauss_length


def get_run_dir():
    # rundir = os.getenv("SLURM_JOB_ID")
    # if rundir is None:
    #     rundir = os.getenv("SLURM_ARRAY_JOB_ID")
    # if rundir is None:
    rundir = get_run_id()
    return rundir


def gauss_length_func(model, ntokens, hidden, init_state, final_state, data, cuda=True):
    """ Generated function to compute gauss map of the logit tangent vector
    """
    def gauss_length(alpha):
        alpha = torch.ones(1) * alpha
        alpha.requires_grad = True
        if cuda:
            alpha = alpha.cuda()

        if hidden is not None:
            z = model.interpolated_forward(data, hidden, alpha, init_state, final_state)[0].squeeze()
        else:
            z = model.interpolated_forward(data, alpha, init_state, final_state).squeeze()
            z = z.view(-1, ntokens)

        # Tangent vector
        v = rop(z, alpha)
        v_norm = torch.norm(v.view(v.shape[0], -1), dim=1, keepdim=True)
        n_v = v / v_norm
        # Acceleration vector
        a = rop(n_v, alpha)

        # Compute acceleration tangent to the velocity
        arclen_a = a - torch.sum((n_v * a).view(v.shape[0], -1), axis=-1, keepdims=True) * n_v
        l_g = torch.sqrt((arclen_a * arclen_a).sum(-1))
        return l_g.detach().cpu().numpy()

    return gauss_length


@ex.automain
def main(model, data, batch_size, clip, epochs, lr, bptt, emsize, nhead, nhid,
         nlayers, dropout, log_interval, alpha_steps, optimizer, cuda, _run):
    corpus = corp.Corpus(data)

    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    eval_batch_size = 10
    train_data = batchify(corpus.train, batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)

    ntokens = len(corpus.dictionary)
    model_name = model
    model = get_model(model, ntokens, emsize, nhead, nhid, nlayers, dropout, device)
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr)
    else:
        raise Exception()
    criterion = nn.CrossEntropyLoss()

    # Loop over epochs.
    best_val_loss = None
    # At any point you can hit Ctrl + C to break out of training early.
    init_state = copy.deepcopy(model.state_dict())
    checkpoint_dir = os.path.join(RUN_DIR, EXPERIMENT_NAME, get_run_dir())
    final_outfile = os.path.join(checkpoint_dir, "init.pt")
    torch.save({
        "model_state": init_state,
    }, final_outfile)

    try:
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train(model, epoch, corpus, clip, lr, train_data, bptt, criterion, log_interval, batch_size, optimizer, _run)
            train_loss = evaluate(model, batch_size, corpus, bptt, criterion, train_data)
            val_loss = evaluate(model, eval_batch_size, corpus, bptt, criterion, val_data)
            print("-" * 89)
            print("| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                  "valid ppl {:8.2f}".format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print("-" * 89)
            # Save the model if the validation loss is the best we"ve seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                for g in optimizer.param_groups:
                    g["lr"] = g["lr"] / 4.0

            _run.log_scalar("train.loss", train_loss)
            _run.log_scalar("train.ppl", math.exp(train_loss))
            _run.log_scalar("train.lr", lr)
            _run.log_scalar("valid.loss", val_loss)
            _run.log_scalar("valid.ppl", math.exp(val_loss))

            _run.log_scalar("train.norm_wdist", param_dist(model.state_dict(), init_state, True))
            _run.log_scalar("train.wdist", param_dist(model.state_dict(), init_state, False))

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")

    final_state = copy.deepcopy(model.state_dict())

    checkpoint_dir = os.path.join(RUN_DIR, EXPERIMENT_NAME, get_run_dir())
    final_outfile = os.path.join(checkpoint_dir, "final.pt")
    torch.save({
        "model_state": final_state,
    }, final_outfile)

    # interp_networks(model_name, ntokens, emsize, nhead, nhid, nlayers, dropout, device, batch_size, eval_batch_size, corpus,
    #                 bptt, criterion, train_data, val_data, init_state, final_state, alpha_steps, cuda, _run)

    print("Computing Gauss Length ...")
    alphas = np.linspace(0, 1, alpha_steps, endpoint=True)
    avg_gl = compute_avg_gauss_length(model, ntokens, eval_batch_size,
                                      init_state, final_state, alphas, bptt,
                                      val_data)
    # try:
    #     alphas = np.linspace(0, 1, alpha_steps, endpoint=True)
    #     avg_gl = compute_avg_gauss_length(model, ntokens, eval_batch_size,
    #                                       init_state, final_state, alphas, bptt,
    #                                       val_data)
    # except:
    #     avg_gl = 0.
    _run.log_scalar("gauss_len", avg_gl)
