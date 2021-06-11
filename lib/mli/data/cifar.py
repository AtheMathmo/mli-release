import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_cifar_10_data(path, train, download, random_augment_train=True):
    if train:
        if random_augment_train:
            transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
            ])
    else:
            transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
              ])
    return datasets.CIFAR10(path, train, download=download, transform=transform)


def get_cifar_100_data(path, train, download, random_augment_train=True):
  if train:
    if random_augment_train:
      transform = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(
              (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
      ])
    else:
      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(
              (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
      ])
  else:
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5071, 0.4867, 0.4408),
                            (0.2675, 0.2565, 0.2761)),
    ])
  return datasets.CIFAR100(path, train, download=download, transform=transform)
