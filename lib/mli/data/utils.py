import scipy.stats as stats
import torch
import numpy as np

from .mnist import get_mnist_data
from .fashionmnist import get_fashion_mnist_data
from .cifar import get_cifar_10_data, get_cifar_100_data


def load_data(dset_name, batchsize, datasize, train=True, shuffle=None, random_augment_train=True):
    dset_map = {
      "fashionmnist": get_fashion_mnist_data,
      "mnist": get_mnist_data,
      "cifar10": get_cifar_10_data,
      "cifar100": get_cifar_100_data
    }
    data = dset_map[dset_name]("./data", train=train, download=True, random_augment_train=random_augment_train)
    if shuffle is None:
       shuffle = train
    if datasize is not None:
        data.data = data.data[:datasize]
        data.targets = data.targets[:datasize]
    return torch.utils.data.DataLoader(data, batch_size=batchsize, shuffle=shuffle)


def corrupt_dataset_labels(loader, random_label_proportion):
    targets = loader.dataset.targets
    use_random_label = torch.LongTensor(stats.bernoulli.rvs(random_label_proportion,
                                                            size=targets.shape[0]).astype(np.long))
    new_targets = use_random_label * torch.randint(0, 10, targets.shape) + (1 - use_random_label) * targets
    loader.dataset.targets = new_targets
