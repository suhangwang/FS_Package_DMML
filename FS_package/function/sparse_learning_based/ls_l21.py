__author__ = 'swang187'

import scipy.linalg as LA
from ...utility.sparse_learning import *


def calculate_obj(X, Y, W, gamma):
    """
    This function calculates the objective function of ls_l21
    """
    temp = np.dot(X, W) - Y
    return np.trace(np.dot(temp.T, temp)) + gamma*calculate_l21_norm(W)


def ls_l21_gradient_descent(X, Y, **kwargs):
    """
    This function implements the least square l21-norm feature selection problem, i.e.,
    ||XW - Y||_2^F + gamma*||W||_{2,1}
    --------------------------
        X: {numpy array}, shape (n_samples, n_features)
            Input data, guaranteed to be a numpy array
        Y: {numpy array}, shape (n_samples, n_classes)
            Each row is a one-hot-coding class label
            guaranteed to be a numpy array
        kwargs: {dictionary}
            gamma: positive scalar
            verbose: boolean
                True if want to display the objective function value
    --------------
    Output:
    --------------
        ind: {numpy array}
    """
    # input error checking
    if 'gamma' not in kwargs:
        print('error, please specify gamma')
        raise
    else:
        gamma = kwargs['gamma']
    if 'max_iter' not in kwargs:
        max_iter = 100
    else:
        max_iter = kwargs['max_iter']
    if 'verbose' not in kwargs:
        verbose = False
    else:
        verbose = kwargs['verbose']

    n_sample, n_feature = X.shape
    xtx = np.dot(X.T, X)    # X^T * X
    D = np.eye(n_feature)
    for i in range(max_iter):
        # update W as W = (X^T X + gamma*D)^-1 (X^T Y)
        temp = LA.inv((xtx + gamma*D))  # (X^T X + gamma*D)^-1
        feature_weights = np.dot(temp, np.dot(X.T, Y))
        # update D as D_ii = 1 / 2 / ||U(i,:)||
        D = generate_diagonal_matrix(feature_weights)

        # display
        if verbose:
            obj = calculate_obj(X, Y, feature_weights, gamma)
            print('obj at iter ' + str(i+1) + ': ' + str(obj) + '\n')
    ind = feature_ranking(feature_weights)
    return ind