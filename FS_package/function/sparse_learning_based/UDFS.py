__author__ = 'swang187'

import scipy
from ...utility.sparse_learning import *
from sklearn.metrics.pairwise import pairwise_distances


def calculate_obj(X, W, M, gamma):
    """
    This function calculate the objective function of ls_l21
    """
    return np.trace(np.dot(np.dot(W.T, M), W)) + gamma*calculate_l21_norm(W)


def construct_M(X, k, gamma):
    """
    This function construct the M matrix described in the paper l2,1-norm
    regularized discriminative feature selection for unsupervised learning
    """
    n_sample, n_feature = X.shape
    Xt = X.T
    D = pairwise_distances(X)
    # sort the distance matrix D in ascending order
    idx = np.argsort(D, axis=1)
    # choose the k-nearest neighbors for each instance
    idx_new = idx[:, 0:k+1]
    H = np.eye(k+1) - 1/(k+1) * np.ones((k+1, k+1))
    I = np.eye(k+1)
    Mi = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        Xi = Xt[:, idx_new[i, :]]
        Xi_tilde =np.dot(Xi, H)
        Bi = np.linalg.inv(np.dot(Xi_tilde.T, Xi_tilde) + gamma*I)
        Si = np.zeros((n_sample, k+1))
        for q in range(k+1):
            Si[idx_new[q], q] = 1
        Mi = Mi + np.dot(np.dot(Si, np.dot(np.dot(H, Bi), H)), Si.T)
    M = np.dot(np.dot(X.T, Mi), X)
    return M


def udfs(X, **kwargs):
    """
    This function implement l2,1-norm regularized discriminative feature
    selection for unsupervised learning, i.e.,
    min_W Tr(W^T M W) + gamma ||W||_{2,1}, s.t. W^T W = I

    Input
    -------------
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a numpy array
    kwargs : {dictionary}
        max_iter: positive integer
        n_clusters: integer
            Number of clusters
        k: integer
            k nearest neighbor
        verbose: boolean
            True if want to display the objective function value
    --------------
    Output
    --------------
    S: {numpy array}, shape(n_samples, d)
        selected features
    """
    # input error checking
        # input error checking
    if 'gamma' not in kwargs:
        print('error, please specify gamma')
        raise
    else:
        gamma = kwargs['gamma']
    if 'k' not in kwargs:
        k = 5
    else:
        k = kwargs['k']
    if 'n_clusters' not in kwargs:
        print('error, please specify the number of clusters: n_clusters')
        raise
    else:
        n_clusters = kwargs['n_clusters']
    if 'max_iter' not in kwargs:
        max_iter = 100
    else:
        max_iter = kwargs['max_iter']
    if 'verbose' not in kwargs:
        verbose = False
    else:
        verbose = kwargs['verbose']

    #construct M
    n_sample, n_feature = X.shape
    M = construct_M(X, k, gamma)

    D = np.eye(n_feature)
    for i in range(max_iter):
        # update W as the eigenvectors of P corresponding to the first n_clusters
        # smallest eigenvalues
        P = M + gamma*D
        eigen_value, eigen_vector = scipy.linalg.eigh(a=P)
        W = eigen_vector[:, 0:n_clusters]
        # update D as D_ii = 1 / 2 / ||W(i,:)||
        D = generate_diagonal_matrix(W)

        # display
        if verbose:
            obj = calculate_obj(X, W, M, gamma)
            print 'obj at iter ' + str(i+1) + ': ' + str(obj)
    return W