import os
import numpy as np
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import kcu as utils
import time
import pandas as pd
from tpot import TPOTClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


""" SETUP """
dry_run = True
cwdir = os.getcwd()
trainfile = cwdir + "/../../data/MNIST/train.csv"
df = pd.read_csv(trainfile)

train_data = df.to_numpy()
train_Y = train_data[:, 0]
train_X = train_data[:, 1:]

# Split train set into train and validation
train_inds, val_inds = sklearn.model_selection.train_test_split(
    np.arange(len(train_Y)), test_size=0.2
)
train_X, val_X = train_X[train_inds], train_X[val_inds]
train_Y, val_Y = train_Y[train_inds], train_Y[val_inds]

if dry_run:
    train_X, train_Y = train_X[:512], train_Y[:512]

""" MAIN CLASSIFICATION PIPELINES """

# Exp01: Several Classifiers including TPOT
utils.boilerplates.run_several_classifiers(train_X, val_X, train_Y, val_Y)

# Exp02: Lets try Pytorch
train_dataset = utils.dataset.MNISTDataset(train_X, train_Y)
val_dataset = utils.dataset.MNISTDataset(val_X, val_Y)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn = utils.models.MNIST_CNN_01().to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
utils.boilerplates.train_classifier(
    cnn, optimizer, train_loader, device, 3, nn.CrossEntropyLoss(), val_loader, show_plot=True
)
