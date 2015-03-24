import numpy as np


def f_score(X, y):
    """
    This function calculates f_score for each feature, where
    f_score = sum((ni/(c-1))*(mean_i - mean)^2)/((1/(n - c))*sum((ni-1)*std_i^2))

    Input
    -----
        X: {numpy array}, shape (n_samples, n_features)
            input data
        y: {numpy array}, shape (n_samples,)
            input class labels

    Output
    ------
        F: {numpy array}, shape (n_features,)
            F-score for each feature
    """

    n_samples, n_features = X.shape
    F = np.zeros(n_features)
    c = np.unique(y)
    mean = np.mean(X)
    for i in range(n_features):
        f = X[:, i]
        mean_list = np.zeros(len(c))
        std_list = np.zeros(len(c))
        num_list = np.zeros(len(c))
        for j in range(len(c)):
            # class_j is a subset of instances with class label c[j]
            class_j = f[y == c[j]]
            mean_list[j] = np.mean(class_j)
            std_list[j] = np.std(class_j)
            num_list[j] = len(class_j)
        t1 = 0
        t2 = 0
        for j in range(len(c)):
            t1 += np.true_divide(num_list[j], len(c)-1) * ((mean_list[j] - mean) ** 2)
            t2 += (num_list[j]-1) * (std_list[j] ** 2)
        t3 = np.true_divide(1, n_samples-len(c))
        F[i] = np.true_divide(t1, t3*t2)
    return F


def feature_ranking(F):
    """
    Rank features in descending order according to F-score, the higher the F-score, the more important the feature is
    """
    idx = np.argsort(F)
    return idx[::-1]
