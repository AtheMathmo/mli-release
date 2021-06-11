import numpy as np
import tqdm

from mli.models import warm_bn, interpolate_state
from .eval import eval_model, eval_model_per_example


def interp_networks(model, init_state, final_state, train_loader, eval_loaders, alpha_steps,
                    loss=None, cuda=True):
    if cuda:
        model.cuda()

    alpha_range = np.linspace(0, 1, alpha_steps, endpoint=True)
    if model.use_batchnorm:
        print('Model uses batchnorm, will take a while...')
    all_metrics = []
    for _ in eval_loaders:
        all_metrics.append({})
    for alpha in tqdm.tqdm(alpha_range):
        interpolate_state(model.state_dict(), init_state, final_state, alpha)
        if model.use_batchnorm:
            warm_bn(model, train_loader, cuda)
        for i, el in enumerate(eval_loaders):
            metrics = eval_model(model, el, loss, cuda)
            for k in metrics:
                if k not in all_metrics[i]:
                    all_metrics[i][k] = []
                all_metrics[i][k].append(metrics[k])
    return alpha_range, all_metrics


def interp_networks_eval_examples(model, init_state, final_state, train_loader, eval_loader,
                                  alpha_steps, cuda):
    if cuda:
        model.cuda()
    all_metrics = {}
    all_logits = []
    alpha_range = np.linspace(0, 1, alpha_steps, endpoint=True)
    if model.use_batchnorm:
        print('Model uses batchnorm, will take a while...')
    for alpha in tqdm.tqdm(alpha_range):
        interpolate_state(model.state_dict(), init_state, final_state, alpha)
        if model.use_batchnorm:
            warm_bn(model, train_loader, cuda)
        metrics, logits = eval_model_per_example(model, eval_loader, cuda)
        for k in metrics:
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(metrics[k])
        all_logits.append(logits)
    return alpha_range, all_metrics, np.array(all_logits)
