import math
from ...utility.sparse_learning import *


def ls_l21_gradient_descent(X, Y, z, **kwargs):
    """
    This function implements the least square l21-norm feature selection problem, i.e.,
    min_{W}||XW - Y||_2^F + gamma*||W||_{2,1}
    --------------------------
    Input
        X: {numpy array}, shape (n_samples, n_features)
            input data, guaranteed to be a numpy array
        Y: {numpy array}, shape (n_samples, n_classes)
            each row is a one-hot-coding class label, guaranteed to be a numpy array
        z: {float}
            regularization parameter
        kwargs: {dictionary}
            verbose: {boolean} true or false
                True if user want to print out the objective function value in each iteration, False if not
    Output:
    --------------
        W: {numpy array}, shape (n_features, n_classes)
            weight matrix
        obj: {numpy array}, shape (n_iterations, )
            objective function value during iterations
    """

    if 'verbose' not in kwargs:
        verbose = False
    else:
        verbose = kwargs['verbose']

    n_sample, n_feature = X.shape
    xtx = np.dot(X.T, X)    # X^T * X
    D = np.eye(n_feature)
    max_iter = 1000
    obj = np.zeros(max_iter)

    for iter_step in range(max_iter):
        # update W as W = (X^T X + gamma*D)^-1 (X^T Y)
        temp = LA.inv((xtx + z*D))  # (X^T X + gamma*D)^-1
        W = np.dot(temp, np.dot(X.T, Y))
        # update D as D_ii = 1 / 2 / ||U(i,:)||
        D = generate_diagonal_matrix(W)
        temp = np.dot(X, W) - Y
        obj[iter_step] = np.trace(np.dot(temp.T, temp)) + z*calculate_l21_norm(W)
        # display
        if verbose:
            print 'obj at iter ' + str(iter_step+1) + ': ' + str(obj[iter_step])
        # determine weather converge
        if iter_step >= 2 and math.fabs(obj[iter_step] - obj[iter_step-1]) < 1e-3:
            break

    return W, obj