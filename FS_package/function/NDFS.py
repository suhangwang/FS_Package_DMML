import scipy.io
import sklearn.cluster
import numpy as np
import sys
from utility.constructW import constructW
from utility.unsupervised_evaluation import evaluation

def kmeansInitialization(X, C):
    """
    This function uses kmeans to initialize the pseudo label
    Input
    ----------
        X: {numpy array}, shape (n_samples, n_features)
            Input data, guaranteed to be a numpy array
        C: {int}
            number of clusters
    Output:
        Y: {numpy array}, shape (n_samples, C)
            pseudo label matrix
    """
    N, d = X.shape
    kmeans = sklearn.cluster.KMeans(n_clusters=C, init='k-means++', n_init=10, max_iter=300, 
                            tol=0.0001, precompute_distances=True, verbose=0, 
                            random_state=None, copy_x=True, n_jobs=1)
    kmeans.fit(X)
    labels = kmeans.labels_
    Y = np.zeros((N,C))
    for row in range(0, N):
        Y[row, labels[row]] = 1
    T = np.dot(Y.transpose(), Y)
    F = np.dot(Y, np.sqrt(np.linalg.inv(T)))
    F = F + 0.02*np.ones((N,C))
    return F

def calculateObj(X,W,F,L,alpha,beta):
    """
    This function calculate the objective function of NDFS
    """
    T1 = np.trace(np.dot(np.dot(F.transpose(),L),F))  # Tr(F^T L F)
    T2 = np.linalg.norm(np.dot(X,W) - F, 'fro')
    T3 = (np.sqrt((W*W).sum(1))).sum()
    obj = T1 + alpha*(T2 + beta*T3)
    return obj
    

def NDFS(X, **kwargs):
    """
    This function implement the NDFS function
    
    Objective Function: 
        min_{F,W} Tr(F^T L F) + alpha*(||XW-F||_F^2 + beta*||W||_{2,1}) + gamma/2 * ||F^T F - I||_F^2
        s.t.    F >= 0
    
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
        maxIter: {int}
            maximal iteration
        verbose: {int} 1 or 0
            1 if user want to print out the objective function value in each iteration, 0 if not
        
    Reference: 
        Li, Zechao, et al. "Unsupervised Feature Selection Using Nonnegative Spectral Analysis." AAAI. 2012. aaaaaaa
    """
    if 'gamma' not in kwargs:
        gamma = 10e8
    else:
        gamma = kwargs['gamma']
    if 'L' not in kwargs:   # L is the Laplacian matrix
        if 'K' not in kwargs:
            K = 5
        else:
            K = kwargs['K']
        # call constructW
        W = constructW(X, weightMode = 'heatKernel', neighborMode = 'knn', k = K)
        L = np.diag(W.sum(1)) - W
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
    if 'maxIter' not in kwargs:
        maxIter = 300
    else:
        maxIter = kwargs['maxIter']
    if 'F0' not in kwargs:
        if 'C' not in kwargs:
            print >>sys.stderr, "either F0 or C should be provided"
        else:
            F = kmeansInitialization(X, kwargs['C'])         # initialize the F
            C = kwargs['C']
    else:
        F = kwargs['F0']
    if 'verbose' not in kwargs:
        verbose = 0
    else:
        verbose = kwargs['verbose']
    
    N, d = X.shape
    D = np.identity(d)  # initialize D as identity matrix
    I = np.identity(N)
    
    count = 0
    obj = np.zeros(maxIter)
    while count < maxIter:
        # update W
        T = np.linalg.inv(np.dot(X.transpose(), X) + beta * D)
        W = np.dot(np.dot(T,X.transpose()),F)
        # update D
        temp = np.sqrt((W*W).sum(1))
        temp[temp<(1e-16)] = 1e-16
        temp = 0.5 / temp
        D = np.diag(temp)
        # update M
        M = L + alpha * (I - np.dot(np.dot(X,T), X.transpose()))
        M = (M + M.transpose())/2
        # update F
        denominator = np.dot(M,F) + gamma*np.dot(np.dot(F,F.transpose()),F)
        temp = np.divide(gamma*F, denominator)
        F = F*np.array(temp)
        temp = np.diag(np.sqrt(np.diag(1 / (np.dot(F.transpose(), F) + 1e-16))))
        F = np.dot(F, temp)
        # calculate objective function
        #obj[count] = calculateObj(X,W,F,L,alpha,beta)
        obj[count] = np.trace(np.dot(np.dot(F.transpose(),M),F)) + gamma/4*np.linalg.norm(np.dot(F.transpose(),F)-np.identity(C), 'fro')
        if verbose:
            print('obj at iter ' + str(count) + ': ' + str(obj[count]) + '\n')
        count = count + 1
    return W, obj

def featureRanking(W):
    """
    Rank featues in descending order according to ||w_i||
    """
    T = (W*W).sum(1)
    ind = np.argsort(T,0)
    return ind[::-1]

def main():
    # load matlab data
    mat = scipy.io.loadmat('data/USPS.mat')
    label = mat['gnd']    # label
    label = label[:,0]
    X = mat['fea']    # data
    N,d = X.shape
    X = X.astype(float)
    # construct W
    #W = sklearn.metrics.pairwise.pairwise_kernels(X, metric='rbf')
    kwargs = {"metric": "euclidean","neighborMode": "knn","weightMode": "heatKernel","k": 5, 't': 1}
    W = constructW(X,**kwargs)
    L = np.diag(W.sum(1)) - W
    
    # feature weight learning / feature selection
    W, obj = NDFS(X, C=40, L=L, verbose=1, maxIter = 30)
    idx = featureRanking(W)
    
    # evalaution
    numFea = 100
    selectedFeatures = X[:,idx[0:numFea]]
    
    ARI, NMI, ACC, predictLabel = evaluation(selectedFeatures = selectedFeatures, C=10, Y=label)
    print ARI
    print NMI
    print ACC
    #print predictLabel.astype(int)
    
if __name__=='__main__':
    main()
