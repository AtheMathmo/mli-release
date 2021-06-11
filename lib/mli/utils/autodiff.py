import torch


def rop(y, x, u=None, retain_graph=True):
    w = torch.ones_like(y, requires_grad=True)
    t = torch.autograd.grad(y, x, w,
                            retain_graph=retain_graph,
                            create_graph=retain_graph,
                            allow_unused=False)[0]
    if u is None:
        u = torch.ones_like(t)
    return torch.autograd.grad(t, w, u,
                               retain_graph=retain_graph,
                               create_graph=retain_graph,
                               allow_unused=False)[0]
