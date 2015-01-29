from sklearn.feature_selection import VarianceThreshold


def low_variance_feature_selection(X, p):
    """
    This function implements the low_variance_feature_selection function (existing methods in scikit-learn)
    Input
    ----------
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a numpy array
    p:{float}
        parameter used to calculate the threshold(threshold = p*(1-p))
    Output
    ----------
    X_new: {numpy array},shape (n_samples, n_features)
        selected features
    """
    sel = VarianceThreshold(threshold=(p * (1 - p)))
    return sel.fit_transform(X)