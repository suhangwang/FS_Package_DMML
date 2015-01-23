import sklearn.cluster
import numpy as np
import sys
from ...utility.construct_W import construct_W


def kmeans_initialization(X, n_clusters):
    """
    This function uses kmeans to initialize the pseudo label
    Input
    ----------
        X: {numpy array}, shape (n_samples, n_features)
            Input data, guaranteed to be a numpy array
        n_clusters: {int}
            number of clusters
    Output:
        Y: {numpy array}, shape (n_samples, C)
            pseudo label matrix
    """
    n_samples, n_features = X.shape
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                                    tol=0.0001, precompute_distances=True, verbose=0,
                                    random_state=None, copy_x=True, n_jobs=1)
    kmeans.fit(X)
    labels = kmeans.labels_
    Y = np.zeros((n_samples, n_clusters))
    for row in range(0, n_samples):
        Y[row, labels[row]] = 1
    T = np.dot(Y.transpose(), Y)
    F = np.dot(Y, np.sqrt(np.linalg.inv(T)))
    F = F + 0.02*np.ones((n_samples, n_clusters))
    return F


def calculate_obj(X, W, F, L, alpha, beta):
    """
    This function calculate the objective function of NDFS
    """
    # Tr(F^T L F)
    T1 = np.trace(np.dot(np.dot(F.transpose(), L), F))
    T2 = np.linalg.norm(np.dot(X, W) - F, 'fro')
    T3 = (np.sqrt((W*W).sum(1))).sum()
    obj = T1 + alpha*(T2 + beta*T3)
    return obj
    

def ndfs(X, **kwargs):
    """
    This function implement the NDFS function
    
    Objective Function: 
        min_{F,W} Tr(F^T L F) + alpha*(||XW-F||_F^2 + beta*||W||_{2,1}) + gamma/2 * ||F^T F - I||_F^2
        s.t. F >= 0
    
    Input:
        X: {numpy array}, shape (n_samples, n_features)
            Input data, guaranteed to be a numpy array
        F0: {numpy array}, shape (n_samples, n_classes)
            initialization of the pseudo label matirx F, if not provided
        L: {numpy array}, shape {n_samples, n_samples}
            Laplacian matrix
        alpha: {float}
            Parameter alpha in objective function
        beta: {float}
            Parameter beta in objective function
        gamma: {float}
            a very large number used to force F^T F = I
        C: {int}
            number of clusters
        max_iter: {int}
            maximal iteration
        verbose: {boolean} True or False
            True if user want to print out the objective function value in each iteration, False if not
        
    Reference: 
        Li, Zechao, et al. "Unsupervised Feature Selection Using Nonnegative Spectral Analysis." AAAI. 2012.
    """
    if 'gamma' not in kwargs:
        gamma = 10e8
    else:
        gamma = kwargs['gamma']
    if 'L' not in kwargs:   # L is the laplacian matrix
        if 'k' not in kwargs:
            k = 5
        else:
            k = kwargs['k']
        # construct affinity matrix W
        W = construct_W(X, weight_mode='heat_kernel', neighborMode='knn', k=k)
        L = np.array(W.sum(1))[:, 0] - W
    else:
        L = kwargs['L']
    if 'alpha' not in kwargs:
        alpha = 1
    else:
        alpha = kwargs['alpha']
    if 'beta' not in kwargs:
        beta = 1
    else:
        beta = kwargs['beta']
    if 'max_iter' not in kwargs:
        max_iter = 300
    else:
        max_iter = kwargs['max_iter']
    if 'F0' not in kwargs:
        if 'n_clusters' not in kwargs:
            print >>sys.stderr, "either F0 or C should be provided"
        else:
            # initialize F
            F = kmeans_initialization(X, kwargs['n_clusters'])
            n_clusters = kwargs['n_clusters']
    else:
        F = kwargs['F0']
    if 'verbose' not in kwargs:
        verbose = 0
    else:
        verbose = kwargs['verbose']
    
    n_samples, n_features = X.shape
    # initialize D as identity matrix
    D = np.identity(n_features)
    I = np.identity(n_samples)
    
    count = 0
    obj = np.zeros(max_iter)
    while count < max_iter:
        # update W
        T = np.linalg.inv(np.dot(X.transpose(), X) + beta * D)
        W = np.dot(np.dot(T, X.transpose()), F)
        # update D
        temp = np.sqrt((W*W).sum(1))
        temp[temp < 1e-16] = 1e-16
        temp = 0.5 / temp
        D = np.diag(temp)
        # update M
        M = L + alpha * (I - np.dot(np.dot(X, T), X.transpose()))
        M = (M + M.transpose())/2
        # update F
        denominator = np.dot(M, F) + gamma*np.dot(np.dot(F, F.transpose()), F)
        temp = np.divide(gamma*F, denominator)
        F = F*np.array(temp)
        temp = np.diag(np.sqrt(np.diag(1 / (np.dot(F.transpose(), F) + 1e-16))))
        F = np.dot(F, temp)
        # calculate objective function

        obj[count] = np.trace(np.dot(np.dot(F.transpose(), M), F)) + gamma/4*np.linalg.norm(np.dot(F.transpose(), F)-np.identity(n_clusters), 'fro')
        if verbose:
            print 'obj at iter ' + str(count) + ': ' + str(obj[count])
        count += 1
    return W, obj


def feature_ranking(W):
    """
    Rank features in descending order according to ||w_i||
    """
    T = (W*W).sum(1)
    idx = np.argsort(T, 0)
    return idx[::-1]
