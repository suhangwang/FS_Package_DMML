import numpy as np
from ...utility.information_gain import *


def feature_ranking(F):
    """
    This function implements the function which sorts features according to their information gain  in a descending order
    Input
    ----------
    F: {numpy array},shape (n_features, )
        F[i] is the information gain of i feature of X
    Output
    ----------
    idx: {numpy array},shape (n_features, )
        index of sorted feature according to their information gain value in descending order
    """
    idx = np.argsort(F)
    return idx[::-1]


def igfs(X, y):
    """
    This function implements the function which selects the feature based on information gain
    Input
    ----------
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete data array
    y : {numpy array}, shape (n_samples, )
        guaranteed to be a numpy array
    Output
    ----------
    F: {numpy array},shape (n_features, )
        F[i] is the information gain of i feature
    """
    n_samples, n_features = X.shape
    # F[i] is the information gain of i feature
    F = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        F[i] = information_gain(f, y)
    return F


