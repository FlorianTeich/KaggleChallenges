"""
Datasets
"""
import os
import pathlib
from zipfile import ZipFile

import torch
import torch.utils.data
import torchvision.transforms as transforms


class DatasetObject:
    """
    Dataset Object
    """

    urls = [""]
    tags = [""]


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
        data_x = self.data[index]
        data_y = self.targets[index]
        if self.transform:
            data_x = self.transform(data_x)
        return data_x, data_y

    def __len__(self):
        return len(self.data)
