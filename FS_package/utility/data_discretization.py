__author__ = 'kewei'
import scipy.io
import numpy as np
import sklearn.preprocessing
def data_discretization(X, num):
    """
    This function implement the data discretization function

    Input
    ----------
    :param X
    :param num
    X: {numpy array}, shape (n_samples, n_features)
        Input data with shape[n_sample,n_features]
    num: Number of samples to generate.Default is 5.

    Output
    ----------
    X_digitized: {numpy array}, shape (n_samples, gini_index)
        output data with shape[n_sample,n_feature] where the features value have been digitized to the range [1,num]
    ----------
    """
    # normalize each feature to the [0,1] range
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    X_normalized = min_max_scaler.fit_transform(X)

    #initialize X_digitized
    n_samples, n_features = X.shape
    X_digitized = (n_samples, n_features)
    X_digitized = np.zeros(X_digitized)

    #digitize data
    bins = np.linspace(0, 1, num)
    for i in range(n_features):
        X_digitized[:, i] = np.digitize(X_normalized[:, i], bins)

    return X_digitized
