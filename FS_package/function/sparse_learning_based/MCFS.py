__author__ = 'swang187'

import scipy
import numpy as np
from sklearn import linear_model
from ...utility.construct_W import construct_W


def mcfs(X, **kwargs):
    """
    This function implements unsupervised feature selection for multi-cluster data

    Input
    -------------
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a numpy array
    kwargs : {dictionary}
        W: {numpy array}, shape (n_sample, n_samples)
            Affinity matrix
        n_clusters: integer
            Number of clusters
        d: integer
            number of feature to select
    --------------
    Output
    --------------
    S: {numpy array}, shape(n_samples, d)
        selected features
    """
    # input error checking
    if 'W' not in kwargs:
        W = construct_W(X)
    else:
        W = kwargs['W']
    if 'n_clusters' not in kwargs:
        print("error, need number of clusters: n_clusters")
        raise
    else:
        n_clusters = kwargs['n_clusters']
    if 'd' not in kwargs:
        print("error, need number of features to be selected: d")
        raise
    else:
        d = kwargs['d']

    # solve the generalized eigen-decomposition problem and get the top K
    # eigen-vectors with respect to the smallest eigenvalues
    W = W.toarray()
    W = (W + W.T) / 2
    W_norm = np.diag(np.sqrt(1 / W.sum(1)))
    W = np.dot(W_norm, np.dot(W, W_norm))
    WT = W.T
    W[W < WT] = WT[W < WT]
    eigen_value, ul = scipy.linalg.eigh(a=W)
    Y = np.dot(W_norm, ul[:, -1*n_clusters-1:-1])

    # solve K L1-regularized regression problem using LARs algorithm with cardinality constraint being d
    n_sample, n_feature = X.shape
    coefficients = np.zeros((n_feature, n_clusters))
    for i in range(n_clusters):
        clf = linear_model.Lars(n_nonzero_coefs=d)
        clf.fit(X, Y[:, i])
        coefficients[:, i] = clf.coef_

    # compute the MCFS score for each feature
    mcfs_score = coefficients.max(1)

    ind = np.argsort(mcfs_score, 0)
    ind = ind[::-1]

    return X[:, ind[0:d]]