import torch


def interpolate_state(model_state, state1, state2, alpha):
    for p_name in model_state:
        if "batches" not in p_name:
            model_state[p_name].zero_()
            model_state[p_name].add_(1.0 - alpha, state1[p_name])
            model_state[p_name].add_(alpha, state2[p_name])


def warm_bn(model, loader, cuda, epochs=1):
    model.reset_bn()
    training = model.training
    model.train()
    with torch.no_grad():
        for _ in range(epochs):
            for x, y in loader:
                if cuda:
                    x, y = x.cuda(), y.cuda()
                _logits = model(x)
    model.train(training)
