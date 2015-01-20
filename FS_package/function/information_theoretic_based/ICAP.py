__author__ = 'kewei'

import numpy as np
from ...utility.entropy_estimators import *


def icap(X, y, **kwargs):
    """
    This function implements the icap function
    The scoring criteria is calculated based on the formula j_icap = I(f;y) - max(0,(I(fj;f)-I(fj;f|y)))
    Input
    ----------
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete numpy array
    y : {numpy array},shape (n_samples, )
        guaranteed to be a numpy array
    kwargs: {dictionary}
        n_selected_features: {int}
            indicates the number of features to select
    Output
    ----------
    F: {numpy array},shape (n_features, )
        Index of selected features, F(1) is the most important feature.
    """
    n_samples, n_features = X.shape
    # F contains the indexes of selected features, F(1) is the most important feature
    F = []
    # is_n_selected_features_specified indicates that whether user specifies the number of features to select
    is_n_selected_features_specified = False
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_n_selected_features_specified = True
    '''
    ee.midd(x,y) is used to estimate the mutual information between discrete variable x and y
    ee.cmidd(x,y,z) is used to estimated the conditional mutual information between discrete variables x and y conditioned on discrete variable z
    '''
    # t1 is a I(f;y) vector for each feature f in X
    t1 = np.zeros(n_features)
    # max is a max(0,(I(fj;f)-I(fj;f|y))) vector for each feature f in X
    max = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)
    while True:
        '''
        we define j_icap as the largest j_icap of all features
        we define idx as the index of the feature whose j_icap is the largest
        j_icap = I(f;y) - max(0,(I(fj;f)-I(fj;f|y)))
        '''
        if len(F) == 0:
            # select the feature whose mutual information is the largest as the first
            idx = np.argmax(t1)
            # put the index of feature whose mutual information is the largest into F
            F.append(idx)
            # f_select is the feature we select
            f_select = X[:, idx]

        if is_n_selected_features_specified is True:
            if len(F) == n_selected_features:
                break
        if is_n_selected_features_specified is not True:
            if j_icap <= 0:
                break

        # we assign an extreme small value to j_icap to make sure that it is smaller than possible value of j_icap
        j_icap = -1000000000000
        for i in range(n_features):
            if i not in F:
                '''
             t2 = I(f;fj), t3 = I(f;fj|y)(fj in F), max[i] is max(0,(I(fj;f)-I(fj;f|y))) for feature i
             '''
                f = X[:, i]
                t2 = midd(f_select, f)
                t3 = cmidd(f_select, f, y)
                if t2-t3 > max[i]:
                    max[i] = t2-t3
                # calculate j_icap for feature i (not in F)
                t = t1[i] - max[i]
                # record the largest j_icap and its index
                if t > j_icap:
                    j_icap = t
                    idx = i
        # put the idx of feature whose j_icap is the largest into F
        F.append(idx)
        # f_select is the feature we select
        f_select = X[:, idx]
    return np.array(F)
