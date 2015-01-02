__author__ = 'kewei'
import numpy as np
import sklearn.preprocessing


def data_discretization(X, n_bins):
    """
    This function implement the data discretization function

    Input
    ----------
        X: {numpy array}, shape (n_samples, n_features)
            Input data, guaranteed to be a numpy array
        num: {int}
            number of bins

    Output
    ----------
        X_digitized: {numpy array}, shape (n_samples, n_features)
            Output data, where features are digitized to n_bins
    """

    # normalize each feature
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    X_normalized = min_max_scaler.fit_transform(X)

    # discretize X
    n_samples, n_features = X.shape
    X_discretized = np.zeros(n_samples, n_features)

    bins = np.linspace(0, 1, n_bins)
    for i in range(n_features):
        X_discretized[:, i] = np.digitize(X_normalized[:, i], bins)

    return X_discretized
