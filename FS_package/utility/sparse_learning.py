__author__ = 'swang187'

import numpy as np


def feature_ranking(W):
    """
    This function rank features according to the feature weights matrix W
    ------------------
    Input:
        W: {numpy array}, shape (n_features, n_clusters)
            feature weights matrix, guaranteed to be a numpy array
    ------------------
    Output:
        ind: {numpy array}, shape {n_features,}
            feature index ranked in descending order of feature importance
    """
    T = (W*W).sum(1)
    ind = np.argsort(T, 0)
    return ind[::-1]


def generate_diagonal_matrix(U):
    """
    This function generate a diagonal matrix D from an input matrix U as D_ii = 1 / 2 / ||U[i,:]||
    ----------------
    Input:
        U: {numpy array}, shape (n, k)
    Output:
        D: {numpy array}, shape (n, n)
    """
    temp = np.sqrt(np.multiply(U, U).sum(1))
    temp[temp < 1e-16] = 1e-16
    temp = 0.5 / temp
    D = np.diag(temp)
    return D

def calculate_l21_norm(X):
    """
    This function calculate the l21 norm of a matrix X, i.e., \sum ||X[i,:]||_2
    ------------
    Input:
        X: {numpy array}, shape (n, k)
    ------------
    Output:
        l21_norm: scalar
    ------------
    """
    return (np.sqrt(np.multiply(X, X).sum(1))).sum()


def construct_label_matrix(label):
    """
    This function convert a 1d numpy array to a 2d array
    -------------
    Input:
        label: {numpy array}, shape(n_sample,)
    ------------
    Output:
        label_matrix: {numpy array}, shape(n_sample, n_class)
    """

    n_samples = label.shape[0]
    unique_label = np.unique(label)
    n_classes = unique_label.shape[0]

    label_matrix = np.zeros((n_samples, n_classes))

    for i in range(n_classes):
        label_matrix[label == unique_label[i], i] = 1

    return label_matrix.astype(int)