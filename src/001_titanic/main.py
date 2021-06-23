import sklearn
from sklearn import tree
import os
import numpy as np
import pandas as pd
import kcu as utils


cwdir = os.getcwd()
trainfile = cwdir + "/../../data/Titanic/train.csv"
testfile = cwdir + "/../../data/Titanic/test.csv"
train_data_pd = pd.read_csv(trainfile)

features = ['Pclass', 'Sex', 'SibSp', 'Parch']
X_train = pd.get_dummies(train_data_pd[features])
Y_train = train_data_pd['Survived']

multi_corr = utils.utils.multiple_correlation(train_data_pd[['Pclass', 'SibSp', 'Parch', "Survived"]], "Survived")

utils.utils.correlation_matrix(pd.DataFrame(X_train))

utils.boilerplates.determine_durations(len(X_train.columns), 10000, sklearn.svm.SVC())

train_inds, val_inds = sklearn.model_selection.train_test_split(
    np.arange(len(Y_train)), test_size=0.2
)
train_X, val_X = X_train.iloc[train_inds], X_train.iloc[val_inds]
train_Y, val_Y = Y_train.iloc[train_inds], Y_train.iloc[val_inds]

utils.boilerplates.run_several_classifiers()
