import os

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image


class DatasetObject:
    urls = [""]
    tags = [""]


class PennDataset:
    """
    The Penn Dataset
    """

    urls = ["https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"]


class MNISTDataset(torch.utils.data.Dataset):
    """
    The MNIST Dataset

    """

    def __init__(
        self,
        data,
        targets,
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=20),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        ),
    ):
        self.data = torch.from_numpy(data.reshape(-1, 1, 28, 28)).float()
        self.targets = torch.LongTensor(torch.from_numpy(targets.astype(int)))
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)
