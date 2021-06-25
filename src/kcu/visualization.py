import os
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import numpy as np


def show_class_distribution(y, relative=True):
    if relative:
        sns.barplot(x=np.unique(y), y=np.unique(y, return_counts=True)[1] / float(len(y)),
                    palette=sns.color_palette("hls", len(np.unique(y))))
    else:
        sns.barplot(x=np.unique(y), y=np.unique(y, return_counts=True)[1], palette=sns.color_palette("hls", len(np.unique(y))))
    plt.title("Class distribution")
    plt.show()
    return


def show_image_from_path(path, hidden=False):
    """
    Show image given its path

    :param path: Path to the actual image file
    :param hidden: Do not actually show image
    :return: Return 1 if image exists and can be displayed, -1 otherwise
    """
    if os.path.exists(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        if hidden:
            plt.show()
        return 1
    else:
        logging.error("No such file exists.")
    return -1
