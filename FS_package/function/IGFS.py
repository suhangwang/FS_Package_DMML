__author__ = 'kewei'
import scipy.io
import numpy as np
import utility.entropy_estimators as ee
import utility.information_gain as ig

def featureRanking(F):
    """
    This function implement the function sort the features according to their information gain value in descending order
    Input
    ----------
    :param F:
    F: {numpy array}
        F[i] is the information gain of i feature of X
    Output
    ----------
    idx: {numpy array}
        index of sorted feature according to their information gain value in descending order
    ----------

    """
    idx = np.argsort(F)
    return idx[::-1]


def igfs(X, y):
    """
    This function implement the function select the feature based on information gain
    Input
    ----------
    :param X:
    :param y:
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete data matrix
    y : {numpy array}, shape (n_samples, label)
        guaranteed to be a numpy array
    Output
    ----------
    F: {numpy array}, shape
        F[i] is the information gain of i feature
    ----------

    """
    n_samples, n_features = X.shape
    # F[i] is the information gain of i feature
    # initialize F
    F = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        F[i] = ig.information_gain(f, y)
    return F


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/iris.mat')
    y = mat['gnd']    # label
    y = y[:,0]
    X = mat['fea']    # data
    X = X.astype(float)

    # feature weight learning / feature selection
    F = igfs(X, y)
    idx = featureRanking(F)
    print F
    print idx

if __name__=='__main__':
    main()




