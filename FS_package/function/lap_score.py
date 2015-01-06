import numpy as np
from scipy.sparse import *
from ..utility.construct_W import construct_W


def feature_select(X, **kwargs):
    """
    This function implement the LapScore function
    1. Construct the weight matrix W if it is not specified
    2. For the r-th feature, we define fr = data(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
    3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
    4. Laplacian score for the r-th feature is Lr = (fr_hat'*L*fr_hat)/*(fr_hat'*D*fr_hat)

    Input
    ----------
        X: {numpy array}, shape (n_samples, n_features)
            Input data, guaranteed to be a numpy array
        kwargs: {dictionary}
            W: {numpy array}, shape (n_samples, n_samples)
            Input weight matrix
    Output
    ----------
        score: {numpy array}, shape (n_features, )
            laplacian score for each feature
    Reference:
        He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS. 2005.
    """

    # if 'W' is not specified, use the default W
    if 'W' not in kwargs.keys():
        W = construct_W(X)
    # construct the affinity matrix W
    W = kwargs['W']
    # build the diagonal D matrix from affinity matrix W
    D = np.array(W.sum(axis=1))
    L = W
    tmp = np.dot(np.transpose(D), X)
    D = diags(np.transpose(D), [0])
    Xt = np.transpose(X)
    t1 = np.transpose(np.dot(Xt, D.todense()))
    t2 = np.transpose(np.dot(Xt, L.todense()))
    # compute the numerator of Lr
    D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # compute the denominator of Lr
    L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # avoid the denominator of Lr to be 0
    D_prime[D_prime < 1e-12] = 10000

    # compute laplacian score for all features
    score = np.array(np.multiply(L_prime, 1/D_prime))[0, :]
    return np.transpose(score)


def feature_ranking(score):
    """
    Rank features in descending order according to fisher score, the higher the laplacian score, the more important the
    feature is
    """
    ind = np.argsort(score, 0)
    return ind[::-1]
