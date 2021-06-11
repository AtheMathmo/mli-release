import torch

import numpy as np
import tqdm

from mli.utils.autodiff import rop
from mli.models import warm_bn, interpolate_state


def trapez_integrate(fn, alphas):
    f0 = fn(alphas[0])
    integral = 0.0
    for i in range(1, len(alphas)):
        f1 = fn(alphas[i])
        integral += (f1 + f0) * 0.5 * (alphas[i] - alphas[i - 1])
        f0 = f1
    return integral


def trapez_integrate_arrays(arr, alphas):
    integral = 0.0
    for i in range(1, len(alphas)):
        f0 = arr[i - 1]
        f1 = arr[i]
        integral += (f1 + f0) * 0.5 * (alphas[i] - alphas[i - 1])
    return integral


def gauss_length_func(model, init_state, final_state, data, cuda=True):
    """ Generated function to compute gauss map of the logit tangent vector
    """
    def gauss_length(alpha):
        alpha = torch.ones(1) * alpha
        alpha.requires_grad = True
        if cuda:
            alpha = alpha.cuda()

        # Compute the interpolated-weight logits
        z = model.interpolated_forward(data, alpha, init_state, final_state).squeeze()

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


def compute_avg_gauss_length(model, init_state, final_state, alphas, loader):
    gauss_length_sum = 0.0
    data_count = 0.0
    model.eval()

    for (x, _) in tqdm.tqdm(loader):
        data_count += x.shape[0]
        x = x.cuda()
        gl = gauss_length_func(model, init_state, final_state, x, True)
        gauss_lengths = trapez_integrate(gl, alphas)
        gauss_length_sum += gauss_lengths.sum()
    avg_gauss_length = gauss_length_sum / data_count
    return avg_gauss_length


def compute_avg_gauss_length_bn(model, init_state, final_state, alphas, train_loader, eval_loader,
                                cuda=True, bn_warm_steps=None):
    dzda = []
    datacount = 0
    for a in tqdm.tqdm(alphas):
        # Need this here for batchnorm warmup
        interpolate_state(model.state_dict(), init_state, final_state, a)
        if model.use_batchnorm:
            warm_bn(model, train_loader, cuda, 1 if bn_warm_steps is None else bn_warm_steps)
        datacount = 0
        model.eval()
        t_dzda = []
        for (x, _) in eval_loader:
            if cuda:
                x = x.cuda()
            gl = gauss_length_func(model, init_state, final_state, x, True)(a)
            t_dzda.append(gl)
            datacount += x.shape[0]
        dzda.append(np.concatenate(t_dzda, 0))
    avg_gl = trapez_integrate_arrays(dzda, alphas).sum() / datacount
    return avg_gl
