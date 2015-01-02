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
    D = np.diag(W.sum(1))
    L = D - W
    #eigen_value, Y = eigs(A=L, M=D, k=n_clusters, which='SM')
    # TODO, whcih one is better, how to deal with complex number
    eigen_value, ul, ur = scipy.linalg.eig(a=L, b=D, left=True)
    ind = np.argsort(eigen_value.real, 0)
    Y = ul[:, ind[0:n_clusters]]

    # solve K L1-regularized regression problem using LARs algorithm with
    # the cardinality constraint set to d
    n_sample, n_feature = X.shape
    coefficients = np.zeros((n_feature, n_clusters))
    for i in range(n_clusters):
        clf = linear_model.Lars(n_nonzero_coefs=d)
        clf.fit(X, Y[:,i].real)
        coefficients[:,i] = clf.coef_

    # compute the MCFS score for each feature
    mcfs_score = coefficients.max(1)

    ind = np.argsort(mcfs_score, 0)
    ind = ind[::-1]

    return X[:,ind[0:d]]


def main():
    # load matlab data
    mat = scipy.io.loadmat('data/COIL20.mat')
    label = mat['gnd']
    label = label[:, 0]
    X = mat['fea']
    n_sample, n_feature = X.shape
    X = X.astype(float)
    #construct W
    kwargs = {"metric": "euclidean","neighborMode": "knn","weightMode": "heatKernel","k": 5, 't': 0.1}
    W = construct_W(X, **kwargs)

    # mcfs feature selection
    numFea = 200
    selected_features = mcfs(X=X, W=W, n_clusters=20, d=numFea)

    # evaluation
    ARI, NMI, ACC, predictLabel = evaluation(selectedFeatures = selected_features, C=20, Y=label)
    print ARI
    print NMI
    print ACC

if __name__=='__main__':
    main()