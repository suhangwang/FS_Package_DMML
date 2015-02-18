import numpy as np
from scipy.stats import norm
from sklearn import linear_model


def alpha_investing(X, y, w0, dw):
    y = y - np.mean(y)
    n_samples, n_features = X.shape
    X = np.transpose(np.transpose(X)-np.transpose(np.tile(np.mean(X, axis=0), [n_samples, 1])))
    sel = np.repeat(np.nan, 0)
    # initials
    res = y - np.mean(y)
    sigma = np.std(res)
    w = w0
    X_sel = np.ones(n_samples)
    clf = linear_model.LinearRegression(fit_intercept=False)
    for i in range(n_features):
        X_can = X[:, i]  # generate next feature
        alpha = w/2/(i+1)
        Xi = X[:, i]
        Xi_new = Xi.reshape(n_samples, 1)
        clf.fit(Xi_new, X_sel)
        r_squared = clf.score(Xi_new, X_sel)
        rho = np.sqrt(1-r_squared)
        # compute corrected t-statistic
        abs = np.abs(np.inner(np.transpose(X_can), res))/np.sqrt(np.inner(np.transpose(X_can), X_can))/rho/sigma
        if np.isnan(abs):
            continue
        pval = 2-2*norm.cdf(abs)
        if pval < alpha:  # compare p-value to threshold
            X_tmp = np.concatenate((X_sel.reshape(n_samples, -1), X_can.reshape(n_samples, 1)), axis=1)
            clf.fit(X_tmp, y)
            if np.logical_not(np.all(np.logical_not(np.isnan(clf.coef_)))):
                continue
            sel = np.append(sel, (i+1))  # add feature to model
            X_sel = X_tmp
            y_predict = clf.predict(X_tmp)
            res = y - y_predict
            sigma = np.std(res)*np.sqrt(n_samples - 1)/np.sqrt(n_samples - 1 - len(sel))
            w = w + dw - alpha
        else:  # otherwise, reject
            w -= alpha  # reduce wealth
        if w <= 0:
            break
    return sel.astype(int)
