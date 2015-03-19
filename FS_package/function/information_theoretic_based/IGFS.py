import numpy as np
from ...utility.information_gain import *


def feature_ranking(F):
    """
    This function implements the function which sorts features according to their information gain in descending order

    Input
    -----
    F: {numpy array}, shape (n_features,)
        F[i] is the information gain value for the i-th feature

    Output
    ------
    idx: {numpy array}, shape (n_features,)
        index of sorted feature according to their information gain value in descending order
    """
    idx = np.argsort(F)
    return idx[::-1]


def igfs(X, y):
    """
    This function implements the function which selects features based on information gain value

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be a discrete data array

    y : {numpy array}, shape (n_samples,)
        input label, guaranteed to be a discrete one-dimensional numpy array

    Output
    ------
    F: {numpy array}, shape (n_features,)
        F[i] is the information gain value for the i-th feature
    """

    n_samples, n_features = X.shape
    F = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        F[i] = information_gain(f, y)
    return F


