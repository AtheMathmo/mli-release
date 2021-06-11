import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os


class ImageNetDataBuilder:
    def __init__(self, datadir, distributed):
        self._distributed = distributed
        
        # Data loading code
        self.traindir = os.path.join(datadir, "train")
        self.valdir = os.path.join(datadir, "val")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.train_dataset = datasets.ImageFolder(
            self.traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        self.val_dataset = datasets.ImageFolder(
            self.valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    def build_sampler_and_loader(self, batch_size, num_workers, train=True):
        dataset = self.train_dataset if train else self.val_dataset
        if train and self._distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset)
        else:
            sampler = None
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=sampler
        )
        return sampler, loader
