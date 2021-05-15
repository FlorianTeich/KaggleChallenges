import sklearn
from sklearn import tree
import os
import numpy as np
import pandas as pd


cwdir = os.getcwd()
trainfile = cwdir + "/../../data/Titanic/train.csv"
testfile = cwdir + "/../../data/Titanic/test.csv"
train_data_pd = pd.read_csv(trainfile)

features = ['Pclass', 'Sex', 'SibSp', 'Parch']
X_train = pd.get_dummies(train_data_pd[features])
Y_train = train_data_pd['Survived']

train_inds, val_inds = sklearn.model_selection.train_test_split(
    np.arange(len(Y_train)), test_size=0.2
)
train_X, val_X = X_train.iloc[train_inds], X_train.iloc[val_inds]
train_Y, val_Y = Y_train.iloc[train_inds], Y_train.iloc[val_inds]

# Approach 1: Decision Trees
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_X, train_Y)
pred = clf.predict(val_X)

print("Decision Tree Acc:", sklearn.metrics.accuracy_score(val_Y, pred))
print("Finished!")
