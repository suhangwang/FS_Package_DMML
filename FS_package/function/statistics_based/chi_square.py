from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def chi_square(X, y, n_selected_features):
    """
    This function implements the chi_square function (existing methods in scikit-learn)
    Input
    ----------
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a numpy array
    y : {numpy array},shape (n_samples, )
        guaranteed to be a numpy array
    n_selected_features : {int}
        indicates the number of features to select
    Output
    ----------
    X_new: {numpy array},shape (n_samples, n_features)
        array of selected features
    """
    return SelectKBest(chi2, k=n_selected_features).fit_transform(X, y)