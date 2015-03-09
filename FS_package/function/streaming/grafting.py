import numpy as np
from scipy.optimize import minimize


def grafting(X, y, gamma):
    w = np.empty(0)
    free = np.empty(0)
    n_samples, n_features = X.shape
    for i in range(n_features):
        Xt = X[:, 0:(i+1)]
        Xty = np.dot(Xt.T, y)
        XtXt = np.dot(Xt.T, Xt)
        yy = np.dot(y.T, y)
        w = np.append(w, 0)
        gradient = np.dot(XtXt, w)-Xty

        # grafting test for new features
        if np.abs(gradient[i]) > gamma:
            free = np.append(free, 1)
        else:
            free = np.append(free, 0)
        if np.sum(free) > 0:
            w_tmp = minimize(lasso_obj, w[free > 0], args=(XtXt[free > 0, free > 0], Xty[free > 0], yy, gamma))
            w = np.zeros((i+1),)
            w[free == 1] = w_tmp.x
            free = np.abs(w) > 0
    return np.nonzero(free)[0]


def lasso_obj(w, XtXt, Xty, yy, gamma):
    """
    This function calculate the objective function of ||Xw-y||_{F}^{2}+\sum_{i}|w_{i}|
    """
    f = np.sum(np.dot(np.dot(w, XtXt), w) - 2*np.dot(w, Xty) + yy) + gamma*np.sum(np.abs(w))
    return f