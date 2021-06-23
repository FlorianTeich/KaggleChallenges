import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image


class DatasetObject():
    urls = [""]
    tags = [""]


class PennDataset():
    urls = ["https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"]


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = torch.from_numpy(data.reshape(-1, 1, 28, 28)).float()
        self.targets = torch.LongTensor(torch.from_numpy(targets.astype(int)))
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)
