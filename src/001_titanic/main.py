import sklearn
from sklearn import tree
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelBinarizer

#PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked


cwdir = os.getcwd()
trainfile = cwdir + "/../../data/Titanic/train.csv"
testfile = cwdir + "/../../data/Titanic/train.csv"
train_data_pd = pd.read_csv(trainfile)
train_data = train_data_pd.to_numpy()

one = OneHotEncoder(handle_unknown='ignore')
ord = OrdinalEncoder(handle_unknown='ignore')
cabins = one.fit_transform(train_data_pd[["Cabin"]])


train_Y = train_data[:, 1]
train_X = train_data[:, 2:]

# Remove name from Data
train_X = np.delete(train_data, 3, axis=1)

#enc.fit(train_X)
#new_data = enc.transform(train_X).toarray()

train_inds, val_inds = sklearn.model_selection.train_test_split(
    np.arange(len(train_Y)), test_size=0.2
)
train_X, val_X = train_X[train_inds], train_X[val_inds]
train_Y, val_Y = train_Y[train_inds], train_Y[val_inds]

# Approach 1: Decision Trees
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_X, train_Y)
pred = clf.predict(val_X, val_Y)

print("Decision Tree Acc:", sklearn.metrics.accuracy_score(val_Y, pred))

print("Finished!...")