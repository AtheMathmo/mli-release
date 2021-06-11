import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_fashion_mnist_data(path, train, download, random_augment_train=True):
    """
    Note: random_augment_train is a dummy variable for API consistency.
    """
    return datasets.FashionMNIST(path, train, download=download, transform=transforms.ToTensor())
