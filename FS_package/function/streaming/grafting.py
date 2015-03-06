import numpy as np
from scipy.optimize import minimize


def grafting(X, y, gamma):
    w = np.empty(0)
    free = np.empty(0)
    n_samples, n_features = X.shape
    for i in n_features:
        Xt = X[:, 0:i]
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
        w_tmp = minimize(lasso_obj, w[free > 0], XtXt[free > 0, free > 0], Xty[free > 0], yy, gamma)
        w = np.zeros((i+1),)
        w[free == 1] = w_tmp
        free = np.abs(w) >= 1e-4

        # grafting test for existing features
        gradient_new = np.dot(XtXt, w)-Xty
        for j in n_features:
            if free[j] > 0 and np.abs(gradient[i]) < gamma:
                free[j] = 0
                w[j] = 0


def lasso_obj(w, XtXt, Xty, yy, gamma):
    f = np.sum(np.dot(np.dot(w, XtXt), w) - 2*np.dot(w, Xty) + yy) + gamma*np.sum(np.abs(w))
    return f