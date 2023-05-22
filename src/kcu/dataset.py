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


class TitanicDataset:
    """
    Titanic Dataset
    """

    def get_dataset(self, path: str):
        """
        Get dataset
        """
        #cwd = str(pathlib.Path(__file__).parent.resolve())
        #print(cwd)
        cmd = "kaggle competitions download -c titanic"
        #os.chdir(cwd + "/../../data/")
        os.makedirs("titanic", exist_ok=True)
        os.system(cmd)
        return


class AmazonReviews:
    """
    Amazon Reviews for Sentiment Analysis
    """

    def get_dataset(self):
        """
        Get dataset
        """
        cwd = str(pathlib.Path(__file__).parent.resolve())
        print(cwd)
        cmd = "kaggle datasets download -d bittlingmayer/amazonreviews"
        os.chdir(cwd + "/../../data/")
        os.makedirs("amazonreviews", exist_ok=True)
        os.system(cmd)
        os.chdir("./amazonreviews/")
        with ZipFile("../amazonreviews.zip", "r") as zip_obj:
            # Extract all the contents of zip file in current directory
            zip_obj.extractall()
        return


class SandP500:
    """
    S&P 500
    """

    def get_dataset(self):
        """
        Get dataset
        """
        cwd = str(pathlib.Path(__file__).parent.resolve())
        print(cwd)
        cmd = "kaggle datasets download -d camnugent/sandp500"
        os.chdir(cwd + "/../../data/")
        os.makedirs("sandp500", exist_ok=True)
        os.system(cmd)
        os.chdir("./sandp500/")
        with ZipFile("../sandp500.zip", "r") as zip_obj:
            # Extract all the contents of zip file in current directory
            zip_obj.extractall()
        return


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
        data_x = self.data[index]
        data_y = self.targets[index]
        if self.transform:
            data_x = self.transform(data_x)
        return data_x, data_y

    def __len__(self):
        return len(self.data)
