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
from matplotlib import pyplot as plt
from sklearn import decomposition
from sklearn.manifold import TSNE
import seaborn as sns


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
    train_X, train_Y = train_X[:1024], train_Y[:1024]

""" DATA EXPLORATION """
# Visualize class distribution
utils.visualization.show_class_distribution(train_Y)

# As we know this is 2D image data, visualize some samples:
plt.imshow(train_X[0].reshape(28, 28), cmap="binary")
plt.axis("off")
plt.show()

# Show an average of multiple instances from same class (class 1)
for i in range(10):
    inds = np.where(train_Y == i)
    acc = np.mean(train_X[inds], axis=0)
    plt.imshow(acc.reshape(28, 28), cmap="binary")
    plt.title("Mean values for instances of Class " + str(i))
    plt.axis("off")
    plt.show()

# Correlation analysis: for each class, lets identify important pixels first:
for i in range(10):
    new_Y = train_Y == i
    corr_coeff = pd.DataFrame(np.hstack([train_X, np.expand_dims(new_Y, 1)])).corr()[-1:].to_numpy()[0, :-1]
    plt.imshow(corr_coeff.reshape(28, 28), cmap="PiYG")
    plt.title("Correlations of pixels with target class " + str(i))
    plt.axis("off")
    plt.show()

# Lets do PCA: identifying a good linear combination of features that maximize the total variance
pca = decomposition.PCA(n_components=2)
view = pca.fit_transform(train_X)
plt.scatter(view[:,0], view[:,1], c=train_Y, alpha=0.5, cmap='Set1')
plt.axis("off")
plt.title("PCA scatter plot")
plt.show()

# Lets have a look at what features/pixels were used the most by our two PCA components:
for i in [0,1]:
    plt.imshow(pca.components_[i].reshape(28,28), cmap="PiYG")
    plt.title("PCA, linear combination for Component " + str(i))
    plt.axis("off")
    plt.show()

# There are a lot of pixels with no to only a bit of weight, so we could do feature selection and simply neglect pixels
# e.g. in the corners of the images

# Next, we visualize the t-SNE embedding. The desirable outcome here would be to have compact non-overlapping
# clusters of classes. This plot might already foreshadow - to some extent - how complicated the classification
# task will be
view = TSNE(n_components=2, random_state=0).fit_transform(train_X)
#plt.figure(figsize=(20,10))
plt.scatter(view[:,0], view[:,1], c=train_Y, alpha=0.5, cmap="Set1")
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title("t-SNE embedding")
plt.show()

""" MAIN CLASSIFICATION PIPELINES """

# Exp01: Several Classifiers using kFold CrossValidation
performances = utils.boilerplates.run_several_classifiers(train_X, train_Y, cv=True)

# Lets report their performances
chart = sns.boxplot(x="method", y="accuracy", data=performances)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title("Accuracy of various methods using 5-fold CV")
plt.tight_layout()
plt.show()

# Also, its interesting to see how well the classifiers do with less data
all_perfs = pd.DataFrame(columns=["method", "balanced_accuracy", "num_samples"])
possible_samples = [128, 256, 512, 1024]
for samples in possible_samples:
    performances = utils.boilerplates.run_several_classifiers(train_X[:samples], train_Y[:samples], cv=True)
    performances["num_samples"] = samples
    all_perfs = all_perfs.append(performances)

chart = sns.lineplot(x="num_samples", y="accuracy", hue="method", data=all_perfs)
chart.set_xticklabels(possible_samples)
plt.title("Effect of train set on accuracy of various methods")
plt.tight_layout()
plt.show()

# Lets train an MLP (as they scale better than SVMs with the amount of training data)
clf = MLPClassifier()
clf.fit(train_X, train_Y)
pred = clf.predict(val_X)
score = sklearn.metrics.accuracy_score(val_Y, pred)
print("MLP Val accuracy: " + str(score))

# Plot confusion matrix of best classifier
sklearn.metrics.plot_confusion_matrix(clf,
                      val_X,
                      val_Y,
                      cmap="Reds")
plt.title("MLP confusion matrix on validation data")
plt.show()

# Exp02: Lets leverage the local spatial relations of the input data and use a 2D CNN
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
