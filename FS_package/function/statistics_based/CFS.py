from ...utility.su_calculation import *


def merit_calculation(X, y):
    """
    This function calculates the merit of set X given the label y

    Input
    ----------
        X: {numpy array}, shape (n_samples, n_features)
            Input data, guaranteed to be a discrete data matrix
        y : {numpy array}, shape (n_samples, )
            guaranteed to be a numpy array
    Output
    ----------
        merits: {float}
            merits is the heuristics 'merit' of a feature subset X
    """
    # merits = (k * rcf)/sqrt(k+k*(k-1)*rff)
    # rcf = (1/k)*sum(su(fi,y)) for all fi in X
    # rff = (1/(k*(k-1)))*sum(su(fi,fj)) for all fi and fj in X
    n_samples, n_features = X.shape
    rff = 0
    rcf = 0
    for i in range(n_features):
        fi = X[:, i]
        rcf += su_calculation(fi, y)
        for j in range(n_features):
            if j > i:
                fj = X[:, j]
                rff += su_calculation(fi, fj)
    rff *= 2
    merits = rcf / np.sqrt(n_features + rff)
    return merits


def cfs(X, y):
    """
    This function uses a correlation based heuristic to evaluate the worth of features which called CFS
    Input
    ----------
        X: {numpy array}, shape (n_samples, n_features)
            Input data, guaranteed to be a discrete data matrix
        y : {numpy array}, shape (n_samples, )
            guaranteed to be a numpy array
    Output
    ----------
        F: {numpy array}
            contains the index of selected features
    """
    n_samples, n_features = X.shape
    F = []
    # M is the list for the merit value
    M = []
    # merit is the maximum merit
    # idx is the corresponding index of feature with maximum merit
    while True:
        merit = -100000000000
        idx = -1
        for i in range(n_features):
            if i not in F:
                F.append(i)
                # t is defined as the merit of F and y
                t = merit_calculation(X[:, F], y)
                if t > merit:
                    merit = t
                    idx = i
                F.pop()
        F.append(idx)
        M.append(merit)
        if len(M) > 5:
            if M[len(M)-1] <= M[len(M)-2]:
                if M[len(M)-2] <= M[len(M)-3]:
                    if M[len(M)-3] <= M[len(M)-4]:
                        if M[len(M)-4] <= M[len(M)-5]:
                            break
    return np.array(F)

