import math
from numpy import linalg as LA
from ...utility.sparse_learning import *


def proximal_gradient_descent(X, Y, z):
    n_samples, n_features = X.shape
    n_samples, n_classes = Y.shape

    print n_features, n_classes

    # At t = 1, initialize line search parameter alpha_minus1 = 1, alpha = 1
    alpha_minus1 = 0
    alpha = 1

    # Initial current step size gamma = 0.1
    gamma = 0.1

    # At t = 1, initialize weight matrix W_minus1 and W to be identity
    W_minus1 = np.zeros((n_features, n_classes))
    W = np.zeros((n_features, n_classes))
    W_plus1 = np.zeros((n_features, n_classes))

    # iterate until convergence
    max_iter = 10000
    count = 0
    obj = np.zeros(max_iter)
    while count < max_iter:
        # line search update V_new
        V = W + (alpha_minus1-1)/alpha*(W - W_minus1)
        W_new_gradient = np.dot(np.dot(np.transpose(X), X), W)-np.dot(np.transpose(X), Y)

        # Iterate to get the suitable step size
        j = 1
        while True:
            # calculate U
            U = V - W_new_gradient/gamma
            # compute W_plus1 according to euclidean projection
            for i in range(n_features):
                if LA.norm(U[i, :]) > z/gamma:
                    W_plus1[i, :] = (1-z/(gamma*LA.norm(U[i, :])))*U[i, :]
                else:
                    W_plus1[i, :] = np.zeros(n_classes)
            # compute F(W_plus1)
            F = LA.norm((np.dot(X, W_plus1)-Y), 'fro')**2 + z*calculate_l21_norm(W_plus1)
            # calculate G
            term1 = LA.norm((np.dot(X, V)-Y), 'fro')**2
            V_gradient = np.dot(np.dot(np.transpose(X), X), V)-np.dot(np.transpose(X), Y)
            term2 = np.trace(np.dot(np.transpose(V_gradient),(W_plus1-V)))
            term3 = gamma/2*(LA.norm((W_plus1-V), 'fro')**2)
            term4 = z*calculate_l21_norm(W_plus1)
            G = term1 + term2 + term3 + term4
            # determine if it meets the Armijo-Goldstein rule
            if F > G:
                gamma *= math.pow(2,j)
            else:
                break
            j += 1
        # update W_minus1 and W
        W_minus1 = W
        W = W_plus1
        # update alpha_minus1 and alpha
        alpha_minus1 = alpha
        alpha = (1+math.sqrt(4*alpha+1))/2

        obj[count] = LA.norm((np.dot(X, W)-Y), 'fro')**2 + z*calculate_l21_norm(W)
        print 'obj at iter ' + str(count) + ': ' + str(obj[count])
        if count >= 1 and (obj[count-1] - obj[count] < 1e-4):
            break
        count += 1
    return W

