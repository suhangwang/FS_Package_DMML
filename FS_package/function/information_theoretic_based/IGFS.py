import numpy as np
from FS_package.utility.mutual_information import *


def igfs(X, y):
    """
    This function implements information gain feature selection

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data

    y : {numpy array}, shape (n_samples,)
        input class labels

    Output
    ------
    F: {numpy array}, shape (n_features,)
        information gain value for each feature
    """

    n_samples, n_features = X.shape
    F = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        F[i] = information_gain(f, y)
    return F


def feature_ranking(F):
    idx = np.argsort(F)
    return idx[::-1]


