import numpy as np


def t_score(X, y):
    """
    This function implements the function which calculates t_score of each feature in X.
    t_score is only used for binary problem
    t_score = (mean0-mean1)/sqrt(((std0^2)/n0)+((std1^2)/n1)))
    Input
    ----------
    X: {numpy array}, shape (n_samples, n_features)
        Input data,guaranteed to be a numpy array
    y : {numpy array}, shape (n_samples, )
        guaranteed to be a numpy array, there are only two value in y
    Output
    ----------
    F: {numpy array}, shape (n_features, )
        T-score for each feature
    """
    n_samples, n_features = X.shape
    F = np.zeros(n_features)
    # define c as the vector of different classes in y
    c = np.unique(y)
    if len(c) == 2:
        for i in range(n_features):
            f = X[:, i]
            # class0 is defined as all instance in f belonging to the first class
            # class1 is defined as all instances in f belonging to the second class
            class0 = f[y == c[0]]
            class1 = f[y == c[1]]
            mean0 = np.mean(class0)
            mean1 = np.mean(class1)
            std0 = np.std(class0)
            std1 = np.std(class1)
            n0 = len(class0)
            n1 = len(class1)
            t = mean0 - mean1
            t0 = np.true_divide(std0**2, n0)
            t1 = np.true_divide(std1**2, n1)
            F[i] = np.true_divide(t, (t0 + t1)**0.5)
    else:
        print('y should be guaranteed to a binary class vector')
    return np.abs(F)


def feature_ranking(F):
    """
    Rank features in descending order according to fisher score, the higher the fisher score, the more important the
    feature is
    """
    idx = np.argsort(F)
    return idx[::-1]

