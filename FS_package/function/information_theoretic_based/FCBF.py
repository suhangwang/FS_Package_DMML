import numpy as np
from ...utility.su_calculation import *


def fcbf(X, y, **kwargs):
    """
    This function implements Fast Correlation Based Filter algorithm
    Input
    ----------
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete numpy array
    y : {numpy array}, shape (n_samples, )
        guaranteed to be a numpy array
    kwargs: {dictionary}
        Parameters for threshold
            delta:
                delta is a parameter for threshold,the default value of delta is 0
    Output
    ----------
    F: {numpy array}, shape (n_features, )
        Index of selected features, F(1) is the most important feature.

    Reference:
        Yu, Lei and Liu, Huan."Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution." ICML 2003.
    """
    n_samples, n_features = X.shape
    if 'delta' in kwargs.keys():
        delta = kwargs['delta']
    else:
        # the default value of delta is 0
        delta = 0
    # t1 is an array for each feature f in X, t1[:,0] stores index of features, t1[:,1] stores su of features
    t1 = np.zeros((n_features, 2))
    for i in range(n_features):
        f = X[:, i]
        t1[i, 0] = i
        t1[i, 1] = su_calculation(f, y)
    """
    s_list is su(f;y) array for feature whose su larger than delta,
    s_list[:,0] is the index of features, s_list[:,1] is su(f;y) of features
    """
    s_list = t1[t1[:, 1] > delta, :]
    # F contains the index of selected features, F(1) is the most important feature.
    F = []
    while len(s_list) != 0:
        # inside s_list, select the largest su
        idx = np.argmax(s_list[:, 1])
        # record the index of feature corresponding to su
        fp = X[:, s_list[idx, 0]]
        # remove both index and su of fp from s_list
        np.delete(s_list, idx, 0)
        # append index of fp to F
        F.append(s_list[idx, 0])
        for i in s_list[:, 0]:
            fi = X[:, i]
            if su_calculation(fp, fi) >= t1[i, 1]:
                # construct the mask for feature whose su is larger than su(fp,y)
                idx = s_list[:, 0] != i
                idx = np.array([idx, idx])
                idx = np.transpose(idx)
                # delete the feature by using the mask
                s_list = s_list[idx]
                length = len(s_list)/2
                s_list = s_list.reshape((length, 2))

    return np.array(F, dtype=int)