from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from main import interpolate_state
from logger import create_exp_name
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lookahead import Lookahead
import functools

import os
import copy
import numpy as np
import csv

class SmallNet(nn.Module):
    def __init__(self, hidden=256):
        super(SmallNet, self).__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def compute_distance(m1, m2):
    distance = 0
    for k in m1:
        assert k in m2
        d = m2[k] - m1[k]
        distance += torch.sum(d * d).item()
    return distance ** 0.5

def compute_dot_product(m1, m2, m0):
    # let u1 = m1 - m0 and u2 = m2 - m0. Computes projection of u2 onto u1.
    # Compute projection of model 2 onto model1 direction (subtracting model0)

    cum_dot_product = 0
    w2_norm = 0
    w1_norm = 0
    for k in m1:
        assert k in m2
        assert k in m0
        w1 = m1[k] - m0[k]
        w2 = m2[k] - m0[k]
        cum_dot_product += torch.sum(w1 * w2).item()
        w2_norm += torch.sum(w2 * w2).item()
        w1_norm += torch.sum(w1 * w1).item()
    # we compute w1_norm squared first
    normalized_projection = cum_dot_product / w2_norm
    cos_theta = (cum_dot_product / (w1_norm ** 0.5)) / (w2_norm ** 0.5)
    return normalized_projection, cos_theta

def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0.
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item() * len(data) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    correct /= len(test_loader.dataset)
    return correct, test_loss

def get_loss(model, dataset, device, loss_fn):
    # TODO: look at loss behaves for each point
    """Returns losses for a given dataset"""
    model.eval()
    losses = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for (data, target) in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_fn(output, target).item()
        losses.append(loss)
    return losses

def interp_networks(dataset, init_state, final_state, alpha_steps, device, hidden, loss_fn, alpha_low=0., alpha_high=1.):
    model = SmallNet(hidden=hidden).to(device)
    alpha_range = np.linspace(alpha_low, alpha_high, alpha_steps, endpoint=True)
    max_losses = np.zeros(len(dataset))
    all_losses = []
    alphas = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    for alpha in alpha_range:
        print(alpha)
        interpolate_state(model.state_dict(), init_state, final_state, alpha)
        # loss = get_loss(model, dataset, device) #TODO: try this instead
        loss = test(model, device, loader, loss_fn)
        # max_losses = np.maximum(loss, max_losses) # for anomaly detection
        alphas.append(alpha)
        all_losses.append(loss)
    return alphas, all_losses, max_losses

def prettify(x):
    return [round(e, 5) for e in x]

def compute_loss(output, target, loss_type='mse'):
    bs, classes = output.shape
    # import ipdb; ipdb.set_trace()
    if loss_type == 'nll':
        output = torch.log(output)
        loss = F.nll_loss(output, target)
    elif loss_type == 'mse':
        targets = torch.zeros_like(output)
        targets[torch.arange(bs), target] = 1
        loss = F.mse_loss(output,targets) * classes
        # import ipdb; ipdb.set_trace()
    else:
        raise NotImplementedError
    return loss

def get_optimizer(optim_name, lr):
    optim_map = {
        'sgd': functools.partial(torch.optim.SGD, lr=lr),
        'adam': functools.partial(torch.optim.Adam, lr=lr),
        # 'lookahead': functools.partial(Lookahead, optimizer=functools.partial(torch.optim.SGD, lr=lr)),
    }
    # optim_map = {
    #     'sgd': functools.partial(torch.optim.SGD, lr=lr, momentum=beta),
    #     'adam': functools.partial(torch.optim.Adam, lr=lr, betas=(beta, 0.999))
    # }
    return optim_map[optim_name]

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST / CIFAR experiments')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.03, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--hidden', type=int, default=256,
                        help='hidden units in network')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--out_dir', type=str, default='test', help='where files are saved')
    parser.add_argument('--loss', type=str, default='nll', help='type of loss')
    # parser.add_argument('--num_remove', type=int, default=6000, help='examples to remove')
    parser.add_argument('--alpha_steps', type=int, default=20, help='number of alpha steps')
    parser.add_argument('--dataset_size', type=int, default=-1, help='size of dataset')
    parser.add_argument('--random_label', type=float, default=0, help='proportion of random labels')

    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # num_remove = args.num_remove

    logged_hparams = ['lr', 'optimizer', 'epochs', 'dataset_size', 'alpha_steps', 'hidden', 'random_label', 'dataset_size']
    exp_dict, exp_name = create_exp_name(args, logged_hparams)

    def corrupt_labels(data, dataset_size, random_label):
        # import ipdb; ipdb.set_trace()
        random_label = min(random_label, dataset_size)
        use_random_label = np.random.choice(dataset_size, size=random_label, replace=False)
        label_mask = torch.zeros(dataset_size)
        label_mask[use_random_label] = 1
        # data.train_labels = label_mask * torch.randint(0, 10, data.train_labels.shape) + (1 - label_mask) * data.train_labels.float()
        # data.train_labels = data.train_labels.long()
        data.targets = label_mask * torch.randint(0, 10, data.targets.shape) + (1 - label_mask) * data.targets.float()
        data.targets = data.targets.long()
        return use_random_label

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    # seems like we need to corrupt first with how dataset works
    dataset_size = len(train_dataset)
    corrupted_labels = []
    if args.random_label > 0:
        corrupted_labels = corrupt_labels(train_dataset, dataset_size, int(args.random_label*dataset_size))

    if args.dataset_size > 0:
        discarded_data = len(train_dataset) - args.dataset_size
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [args.dataset_size, discarded_data])
        dataset_size = args.dataset_size


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    print("using device...", device)
    model = SmallNet(args.hidden).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    if args.optimizer == "lookahead":
        args.optimizer = "sgd"
        optimizer = get_optimizer(args.optimizer, args.lr)(model.parameters())
        optimizer = Lookahead(optimizer)
    else:
        optimizer = get_optimizer(args.optimizer, args.lr)(model.parameters())

    # exp_desc =  f"epochs={args.epochs},lr={args.lr},hidden={args.hidden}"
    out_dir = os.path.join("out", args.out_dir)
    out_dir = os.path.join(out_dir, exp_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    init_file = os.path.join(out_dir, "init.pt") # todo: save these somewhere else so we can parallelize
    final_file = os.path.join(out_dir, "final.pt")
    iterations = []
    iteration_losses = []
    iteration_checkpoints = []
    iteration_projections = []
    iteration_cosines = []

    loss_fn = functools.partial(compute_loss, loss_type=args.loss)

    # initial training
    init_state = copy.deepcopy(model.state_dict())
    torch.save(init_state, init_file)
    i = 0
    iterations_per_epoch = int(dataset_size / args.batch_size)
    save_freq = min(50, iterations_per_epoch)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        correct = 0.
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            train_loss += loss.item() * len(data)

            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            if i % save_freq == 0:
                iterations.append(i)
                ckpt_name = os.path.join(out_dir, f"ckpt{i}.pt")
                curr_state = copy.deepcopy(model.state_dict())
                torch.save(curr_state, ckpt_name)
                iteration_checkpoints.append(ckpt_name)
                loss = test(model, device, train_loader, loss_fn) # can also consider looking at train loss
                iteration_losses.append(loss)
                model.train()
            i += 1
        train_loss /= dataset_size
        train_acc = correct / dataset_size
        # test_acc, test_loss = test(model, device, test_loader)
    final_state = copy.deepcopy(model.state_dict())
    for ckpt_name in iteration_checkpoints:
        middle_state = torch.load(ckpt_name)
        projection, cos_theta = compute_dot_product(middle_state, final_state, init_state)
        iteration_projections.append(projection)
        iteration_cosines.append(cos_theta)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iterations, iteration_projections, label="projections")
    ax.plot(iterations, iteration_cosines, label="cosines")
    ax.plot(iterations, [x[1] for x in iteration_losses], label="train losses")
    plt.legend()
    ax.set_xlabel("iterations")
    ax.set_title(exp_name)
    fig.savefig(os.path.join(out_dir, "training.png"))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iterations, [x[1] for x in iteration_losses], label="train losses")
    plt.legend()
    ax.set_xlabel("iterations")
    ax.set_title(exp_name)
    fig.savefig(os.path.join(out_dir, "train_loss.png"))

    alphas, losses, _ = interp_networks(train_dataset, init_state, final_state, args.alpha_steps, device, args.hidden, loss_fn)
    fig, ax = plt.subplots(figsize=(10, 5))
    distance_traveled = compute_distance(init_state, final_state)
    distances = [a * distance_traveled for a in alphas]
    ax.plot(distances, [x[1] for x in losses])
    ax.set_xlabel("distance in interpolation direction")
    ax.set_ylabel("loss")
    ax.set_title(exp_name)
    fig.savefig(os.path.join(out_dir, "interpolation.png"))

    # check if strictly monotonic
    curr_loss = np.inf
    monotonic = "true"
    for x in losses:
        if x[1] < curr_loss:
            curr_loss = x[1]
        else:
            monotonic = "false"
            break
    with open(os.path.join(out_dir, "monotonic.txt"), "w") as f:
        f.write(monotonic)


    torch.save(final_state, final_file)
    import sys; sys.exit(0)

   # prune randomly


if __name__ == '__main__':
    main()