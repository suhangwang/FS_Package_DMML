__author__ = 'kewei'

from ...utility.entropy_estimators import *
from ...utility.conditional_entropy import *


def disr(X, y, **kwargs):
    """
    This function implement the disr function
    The scoring criteria is calculated based on the formula j_disr = sum(I(fk,j;y)/H(f fj y))
    Input
    ----------
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete numpy array
    y : {numpy array},shape (n_samples, )
        guaranteed to be a numpy array

    kwargs: {dictionary}
        n_selected_features:{int}
            indicates the number of features to select
    Output
    ----------
    F: {numpy array},shape (n_features, )
        index of selected features, F(1) is the most important feature.
    """
    n_samples, n_features = X.shape
    # F contains the indexes of selected features, F(1) is the most important feature
    F = []
    # is_n_selected_features_specified indicates that whether user specifies the number of features to select
    is_n_selected_features_specified = False
    '''
    midd(x,y) is used to estimate the mutual information between discrete variable x and y
    cmidd(x,y,z) is used to estimated the conditional mutual information between discrete variables x and y conditioned on discrete variable z
    '''
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_n_selected_features_specified = True

    # sum is a sum(I(fk,j;y)/H(f fj y)) vector for each feature f in X
    sum = np.zeros(n_features)
    while True:
        '''
        we define j_disr as the largest j_disr of all features
        we define idx as the index of the feature whose j_disr is the largest
        j_disr = sum(I(fk,j;y)/H(f fj y))
        '''
        if len(F) == 0:
            # t1 is a I(f;y) vector for each feature f in X
            t1 = np.zeros(n_features)
            for i in range(n_features):
                f = X[:, i]
                t1[i] = ee.midd(f, y)
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
            if j_disr <= 0:
                break

        # we assign an extreme small value to j_disr to make sure that it is smaller than possible value of j_disr
        j_disr = -1000000000000
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                # t1 =I(fk,j;y) =I(fj;y)+I(f;y|fj)
                t1 = ee.midd(f_select, y) + ee.cmidd(f, y, f_select)
                # t2 = H(f fj y)=H(f) + H(fj|f) + H(y|f,fj)
                # H(y|f,fj) = H(y|fj)-I(y;f|fj)
                t2 = ee.entropyd(f) + conditional_entropy(f_select, f) + (conditional_entropy(y, f_select) -
                                                                          ee.cmidd(y, f, f_select))
                sum[i] += np.true_divide(t1, t2)
                # record the largest j_disr and its index
                if sum[i] > j_disr:
                    j_disr = sum[i]
                    idx = i
        # put the index of feature whose j_disr is the largest into F
        F.append(idx)
        # f_select is the feature we select
        f_select = X[:, idx]
    return np.array(F)

