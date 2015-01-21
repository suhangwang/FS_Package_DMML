import numpy as np
from numpy import linalg as LA
from ...utility.sparse_learning import *
from __future__ import division


def proximal_gradient_descent(X, Y, z):
    n_samples, n_features = X.shape
    n_samples, n_classes = Y.shape

    print n_features, n_classes
    # Initial line search parameter alpha = 1
    alpha = 1

    # Initial step size gamma = 0.1
    gamma = 0.01

    # Initial weight matrix W to be identity
    W = np.ones((n_features, n_classes))

    # Iterate until convergence
    while True:
        # Iterate to get the suitable step size
        while True:
            F = LA.norm((np.dot(X, W)-Y), 'fro')**2 + z*calculate_l21_norm(W)
            # calculate U
            U = W - np.dot(np.dot(np.transpose(X), X), W)-np.dot(np.transpose(X), Y)/gamma
            W_new = np.zeros((n_features, n_classes))
            for i in range(n_features):
                if LA.norm(U[i, :]) > z/gamma:
                    W_new[i, :] = (1-z/(gamma*LA.norm(U[i, :])))*U[i, :]
                else:
                    W_new[i, :] = np.zeros(n_classes)
            # calculate G
            term1 = LA.norm((np.dot(X, W)-Y), 'fro')**2 + z*calculate_l21_norm(W)
            term2 = np.trace(np.dot(np.dot(np.transpose(X), X), W)-np.dot(np.transpose(X), Y)*np.transpose(W_new-W))
            term3 = gamma/2*(LA.norm((W_new-W), 'fro')**2)
            term4 = z*calculate_l21_norm(W_new)
            G = term1 + term2 + term3 + term4
            if F > G:
                gamma *= 2
            else:
                break
        # calculate new W

    print F
    return X

