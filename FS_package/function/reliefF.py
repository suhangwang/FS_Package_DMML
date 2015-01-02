import scipy.io
import numpy as np
from utility.construct_W import construct_W
from utility.supervised_evaluation import *


def reliefF(X, y):
    """
    This function implement the reliefF function
    1. Construct the weight matrix W in reliefF way
    2. For the r-th feature, we define fr = X(:,r), reliefF score for the r-th feature is -1+fr'*K*fr

    Input
    ----------
        X: {numpy array}, shape (n_samples, n_features)
            input data, guaranteed to be a numpy array
        y: {numpy array}, shape (n_samples, 1)
            true labels, guaranteed to be a numpy array
    Output
    ----------
        score: {numpy array}, shape (n_features, 1)
            reliefF score for each feature
    """
    kwargs = {"neighbor_mode": "supervised", "reliefF": True, 'y': y}
    W = construct_W(X, **kwargs)
    n_samples, n_features = X.shape
    score = np.zeros(n_features)
    for i in range(n_features):
        score[i] = -1 + np.dot(np.transpose(X[:, i]), W.dot(X[:, i]))
    return score


def feature_ranking(score):
    """
    Rank features in descending order according to reliefF score, the higher the fisher score, the more important the
    feature is
    """
    ind = np.argsort(score, 0)
    return ind[::-1]


def main():
    # load data
    mat = scipy.io.loadmat('../data/ORL.mat')
    label = mat['gnd']    # label
    label = label[:, 0]
    X = mat['fea']    # data
    X = X.astype(float)

    # feature weight learning / feature selection
    score = reliefF(X, label)
    idx = feature_ranking(score)
    # evaluation
    n_features = 100
    selected_features = X[:, idx[0:n_features]]

    acc = evaluation_split(selected_features=selected_features, y=label)
    print acc

if __name__ == '__main__':
    main()