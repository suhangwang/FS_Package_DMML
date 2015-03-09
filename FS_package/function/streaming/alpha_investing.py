import numpy as np
from sklearn import linear_model


def alpha_investing(X, y, w0, dw):
    n_samples, n_features = X.shape
    w = w0
    F = []  # selected features
    for i in range(n_features):
        x_can = X[:, i]  # generate next feature
        alpha = w/2/(i+1)
        X_old = X[:, F]
        if i is 0:
            X_old = np.ones((n_samples, 1))
            linreg_old = linear_model.LinearRegression()
            linreg_old.fit(X_old, y)
            error_old = 1 - linreg_old.score(X_old, y)
        if i is not 0:
            # model built with only X_old
            linreg_old = linear_model.LinearRegression()
            linreg_old.fit(X_old, y)
            error_old = 1 - linreg_old.score(X_old, y)

        # model built with X_old & {x_can}
        X_new = np.concatenate((X_old, x_can.reshape(n_samples, 1)), axis=1)
        logreg_new = linear_model.LinearRegression()
        logreg_new.fit(X_new, y)
        error_new = 1 - logreg_new.score(X_new, y)

        # calculate p-value
        pval = np.exp(-(error_old - error_new)/(2*error_old/n_samples))
        if pval < alpha:
            F.append(i)
            w = w + dw - alpha
        else:
            w -= alpha
    return np.array(F)
