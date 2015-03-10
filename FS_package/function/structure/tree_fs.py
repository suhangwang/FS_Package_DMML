import math
import numpy as np


def group_fs(X, y, z1, z2, idx, **kwargs):
    """
    This function implements supervised sparse group feature selection with least square loss, i.e.,
    min_{w} ||Xw-Y||_2^2 + z_1||x||_1 + z_2*sum_j w_j||w_{G_{j}}||
    --------------------------
    Input
        X: {numpy array}, shape (n_samples, n_features)
            input data, guaranteed to be a numpy array
        Y: {numpy array}, shape (n_samples, n_classes)
            each row is a one-hot-coding class label, guaranteed to be a numpy array
        z1: {float}
            regularization parameter of L1 norm for each element
        z2: {float}
            regularization parameter of L2 norm for the non-overlapping group
        idx: {numpy array}, shape (3, n_nodes)
            3*nodes matrix, where nodes denotes the number of nodes of the tree
            idx(1,:) contains the starting index
            idx(2,:) contains the ending index
            idx(3,:) contains the corresponding weight (w_{j})
        kwargs : {dictionary}
            verbose: {boolean} True or False
                True if user want to print out the objective function value in each iteration, False if not
    --------------------------
    Output
        w: {numpy array}, shape (n_features, )
            weight matrix
        obj: {numpy array}, shape (n_iterations, )
            objective function value during iterations
        value_gamma: {numpy array}, shape (n_iterations, )
            suitable step size during iterations

    Reference:
        Liu, Jun, et al. "Moreau-Yosida Regularization for Grouped Tree Structure Learning." NIPS. 2010.
    """

    if 'verbose' not in kwargs:
        verbose = False
    else:
        verbose = kwargs['verbose']

    # Starting point initialization #
    n_samples, n_features = X.shape
    # compute X'y
    Xty = np.dot(np.transpose(X), y)
