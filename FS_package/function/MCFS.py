__author__ = 'swang187'

import scipy.io
import numpy as np
from scipy.sparse.linalg import eigs
from sklearn import linear_model
from utility.construct_W import construct_W
from utility.unsupervised_evaluation import evaluation

def mcfs(X, **kwargs):
    """
    This function implement unsupervised feature selection for multi-cluster data

    Input
    -------------
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a numpy array
    kwargs : {dictionary}
        W: {numpy array}, shape (n_sample, n_samples)
            Affinity matrix
        n_clusters: integer
            Number of custers
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
        print("error, need affinity matrix: W")
        raise
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


    # solve K L1-regularized regression problem using LARs algorithm with
    # the cardinality constraint set to d
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


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    label = mat['gnd']
    label = label[:, 0]
    X = mat['fea']
    n_sample, n_feature = X.shape
    X = X.astype(float)
    #construct W
    kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 0.1}
    W = construct_W(X, **kwargs)

    # mcfs feature selection
    num_fea = 200
    selected_features = mcfs(X=X, W=W, n_clusters=20, d=num_fea)

    # evaluation
    ari, nmi, acc = evaluation(selected_features=selected_features, n_clusters=20, y=label)
    print ari
    print nmi
    print acc

if __name__ == '__main__':
    main()