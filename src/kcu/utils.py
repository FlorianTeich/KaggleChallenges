import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def correlation_matrix(X):
    plt.rcParams["figure.figsize"] = (12, 6)
    fig, ax = plt.subplots()
    corr_ = X.corr()
    mask = np.triu(np.ones_like(corr_.to_numpy(), dtype=bool))
    sns.heatmap(corr_,
              annot=True,
              linewidths=.5,
              fmt=".2f",
              ax=ax,
              mask=mask,
              cmap='RdBu',
              vmin=-1.0,
              vmax=1.0)

    ax.set_title("Feature correlations")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, horizontalalignment='right')
    plt.show()


def correlation_for_y(X, y, feats_per_line=25):
    plt.rcParams["figure.figsize"] = (12, 6)
    feats = len(X.columns)
    for i in y.unique():
        fig, ax = plt.subplots()
        corr_ = pd.concat([X, y], axis=1, ignore_index=True).corr()[-1][0:-1].to_numpy().reshape(-1, feats_per_line)
        im1 = ax.imshow(corr_, cmap='PiYG', interpolation='nearest', vmin=-1.0, vmax=1.0)
        ax.set_title("Class " + str(i) + " feature correlations.")
        for j in range(feats_per_line):
            for k in range(int(feats / feats_per_line) + 1):
                if (j + (k * 25)) < feats:
                    text = ax.text(j, k, j + (k * feats_per_line),
                               ha="center", va="center", color="gray")
        fig.colorbar(im1, orientation="horizontal")
        ax.set_axis_off()
        #plt.savefig("correlations_" + str(i) + ".svg")
        plt.show()
