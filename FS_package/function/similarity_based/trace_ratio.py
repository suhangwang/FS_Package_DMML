import numpy as np
from scipy.sparse import *
from ...utility.construct_W import construct_W


def trace_ratio(X, y, **kwargs):
    """
    This function implements the trace ratio criterion for feature selection

    Input
    ----------
        X: {numpy array}, shape (n_samples, n_features)
            Input data, guaranteed to be a numpy array
        y: {numpy array}, shape (n_samples, )
            True labels
        kwargs: {dictionary}
            style: {string}
                style == 'fisher', build between-class matrix and within-class affinity matrix in a fisher score way
                style == 'laplacian', build between-class matrix and within-class affinity matrix in a fisher score way
    Output
    ----------
        feature_idx: {numpy array}, shape (n_features, )
            the ranked (descending order) feature index based on subset-level score
        feature_score: {numpy array}, shape (n_features, )
            the feature-level score
        subset_score: {float}, the subset-level score

    Reference:
        Feiping Nie et al. "Trace Ratio Criterion for Feature Selection." AAAI 2008.
    """

    # if 'style' is not specified, use the fisher score way to built two affinity matrix
    if 'style' not in kwargs.keys():
        kwargs['style'] = 'fisher'
    # get the way to build affinity matrix, 'fisher' or 'laplacian'
    style = kwargs['style']
    n_samples, n_features = X.shape

    if style is 'fisher':
        kwargs_within = {"neighbor_mode": "supervised", "fisher_score": True, 'y': y}
        # build within class and between class laplacian matrix L_w and L_b
        W_within = construct_W(X, **kwargs_within)
        L_within = np.eye(n_samples) - W_within
        L_tmp = np.eye(n_samples) - np.ones([n_samples, n_samples])/n_samples
        L_between = L_within - L_tmp

    if style is 'laplacian':
        kwargs_within = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
        # build within class and between class laplacian matrix L_w and L_b
        W_within = construct_W(X, **kwargs_within)
        D_within = np.diag(np.array(W_within.sum(1))[:, 0])
        L_within = D_within - W_within
        W_between = np.dot(np.dot(D_within, np.ones([n_samples, n_samples])), D_within)/np.sum(D_within)
        D_between = np.diag(np.array(W_between.sum(1)))
        L_between = D_between - W_between

    # build X'*L_within*X and X'*L_between*X
    L_within = (np.transpose(L_within) + L_within)/2
    L_between = (np.transpose(L_between) + L_between)/2
    S_within = np.array(np.dot(np.dot(np.transpose(X), L_within), X))
    S_between = np.array(np.dot(np.dot(np.transpose(X), L_between), X))
    S_within = (np.transpose(S_within) + S_within)/2
    S_between = (np.transpose(S_between) + S_between)/2

