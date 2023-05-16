"""
Boilerplate Code
"""
import time
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# try:
#    import cuml
# except:
#    pass


def train_classifier(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    loss_fct,
    val_loader: torch.utils.data.DataLoader,
    show_plot: bool,
):
    """
    Train a torch model and evaluate on validation data

    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        train_loader (_type_): _description_
        device (_type_): _description_
        epochs (_type_): _description_
        loss_fct (_type_): _description_
        val_loader (_type_): _description_
        show_plot (bool): _description_
    """
    test_history_acc = []
    train_history_loss = []
    test_history_loss = []
    train_loss = 0.0
    for epoch in range(epochs):
        model.train()
        for _, (data_x, data_y) in enumerate(train_loader):
            data_x, data_y = data_x.to(device), data_y.to(device)
            pred = model(data_x)
            loss = loss_fct(pred, data_y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader.dataset)  # type: ignore
        train_history_loss.append(train_loss)

        # EVALUATION
        model.eval()
        test_loss, correct = 0.0, 0.0
        for _, (data_x, data_y) in enumerate(val_loader):
            data_x, data_y = data_x.to(device), data_y.to(device)
            pred = model(data_x)
            test_loss += loss_fct(pred, data_y).item()
            correct += (pred.argmax(1) == data_y).type(torch.float).sum().item()

        test_loss /= len(val_loader.dataset)  # type: ignore
        correct /= len(val_loader.dataset)  # type: ignore
        test_history_loss.append(test_loss)
        test_history_acc.append(correct)

        print(
            f"Epoch: {epoch} \t Test Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}"
        )

    if show_plot:
        plt.plot(np.arange(epoch + 1), test_history_acc)
        plt.plot(np.arange(epoch + 1), train_history_loss)
        plt.plot(np.arange(epoch + 1), test_history_loss)
        plt.legend(["test_acc", "train_loss", "test_loss"])
        plt.show()


def determine_durations(
    model,
    try_samples,
    n_features=10,
    n_samples=10000,
):
    """
    Create artificial dataset and train given model for different dataset
    sizes to evaluate training duration

    Args:
        n_features (_type_): _description_
        n_samples (_type_): _description_
        model (_type_): _description_
        try_samples (list, optional): _description_. Defaults to [10, 100, 1000, 2000, 4000, 8000].
    """
    if not try_samples:
        try_samples = [10, 100, 1000, 2000, 4000, 8000]
    x_art, y_art = datasets.make_classification(
        n_samples=n_samples, n_features=n_features
    )
    times = []
    samples = try_samples
    for i in samples:
        start = timer()
        model.fit(x_art[:i], y_art[:i])
        end = timer()
        times.append(end - start)
        print(i, end - start)
    sns.lineplot(x=samples, y=times, marker="o")
    plt.title("Training time for different dataset sizes")
    plt.xlabel(samples)
    plt.ylabel("Training time in seconds")
    plt.show()


def time_classifiers(train_x, train_y, samples):
    """
    Run various classifiers on dataset with different sample sizes to evaluate training duration

    Args:
        train_x (_type_): _description_
        train_y (_type_): _description_
        samples (list, optional): _description_. Defaults to [1000, 2000, 4000, 8000].

    Returns:
        _type_: _description_
    """
    if not samples:
        samples = [1000, 2000, 4000, 8000]
    performances = pd.DataFrame(columns=["method", "time", "samples"])
    methods = [
        ("kNN", sklearn.neighbors.KNeighborsClassifier(5)),
        ("SVM", sklearn.svm.SVC(kernel="linear", C=0.025)),
        ("MLP", MLPClassifier(alpha=1, max_iter=10)),
        ("DecisionTree", sklearn.tree.DecisionTreeClassifier(max_depth=5)),
        (
            "RandomForest",
            sklearn.ensemble.RandomForestClassifier(max_depth=5, n_estimators=10),
        ),
        ("AdaBoost", sklearn.ensemble.AdaBoostClassifier()),
    ]

    for name, clf in methods:
        for sample in samples:
            scaler = sklearn.preprocessing.StandardScaler()
            pipeline = Pipeline([("transformer", scaler), ("estimator", clf)])
            start_time = time.time()
            pipeline.fit(train_x[:sample], train_y[:sample])
            elapsed_time = time.time() - start_time
            performances = performances.append(
                {"method": name, "time": elapsed_time, "samples": sample},
                ignore_index=True,
            )

    return performances


def run_several_classifiers(
    train_x,
    train_y,
    val_x=None,
    val_y=None,
    use_gpu_methods=False,
    crossvalidate=True,
    scoring="accuracy",
):
    """
    Run multiple classifiers on the training set and evaluate on validation set

    Args:
        train_x (_type_): _description_
        train_y (_type_): _description_
        val_x (_type_, optional): _description_. Defaults to None.
        val_y (_type_, optional): _description_. Defaults to None.
        use_gpu_methods (bool, optional): _description_. Defaults to False.
        crossvalidate (bool, optional): _description_. Defaults to True.
        scoring (str, optional): _description_. Defaults to "accuracy".

    Returns:
        _type_: _description_
    """
    performances = pd.DataFrame(columns=["method", scoring])
    methods = [
        ("kNN", sklearn.neighbors.KNeighborsClassifier(5)),
        ("SVM", sklearn.svm.SVC(kernel="linear", C=0.025)),
        ("MLP", MLPClassifier(alpha=1, max_iter=10)),
        ("DecisionTree", sklearn.tree.DecisionTreeClassifier(max_depth=5)),
        (
            "RandomForest",
            sklearn.ensemble.RandomForestClassifier(max_depth=5, n_estimators=10),
        ),
        ("AdaBoost", sklearn.ensemble.AdaBoostClassifier()),
    ]

    if use_gpu_methods:
        methods = [
            # (cuml.neighbors.KNeighborsClassifier(n_neighbors=10), "kNN"),
            # (cuml.ensemble.RandomForestClassifier(), "Random Forest"),
            (XGBClassifier(tree_method="gpu_hist", verbosity=0), "XGBoost"),
        ]

    scorer = sklearn.metrics.get_scorer(scoring)
    for name, clf in methods:
        if not crossvalidate:
            scaler = sklearn.preprocessing.StandardScaler()
            pipeline = Pipeline([("transformer", scaler), ("estimator", clf)])
            pipeline.fit(train_x, train_y)
            pred = pipeline.predict(val_x)
            score = scorer(val_y, pred)
            print(name + " score:", score)
            performances = performances.append(
                {"method": name, scoring: score}, ignore_index=True
            )
        else:
            scaler = sklearn.preprocessing.StandardScaler()
            pipeline = Pipeline([("transformer", scaler), ("estimator", clf)])
            scores = cross_val_score(
                pipeline, train_x, train_y, cv=5, scoring=scoring, n_jobs=1
            )
            print(name + " scores: ", scores)
            for score in scores:
                performances = performances.append(
                    {"method": name, scoring: score}, ignore_index=True
                )
    return performances
