import torch


def param_dist(param1, param2, normalized=False):
    distance = 0.
    if normalized:
        p1_norm = 0
    else:
        p1_norm = 1
    for k in param1:
        assert k in param2
        # Don't count the batchnorm running stats
        if "running" not in k and "batches" not in k:
            d = param1[k] - param2[k]
            distance += torch.sum(d * d).item()
            if normalized:
                p1_norm += torch.sum(param1[k] * param1[k]).item()
    if normalized:
        p1_norm = p1_norm ** 0.5
    return distance ** 0.5 / p1_norm
