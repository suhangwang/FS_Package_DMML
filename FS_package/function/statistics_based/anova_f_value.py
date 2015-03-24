from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


def anova_f_value(X, y, n_selected_features):
    """
    This function implements the anova f_value feature selection (existing method for classification in scikit-learn)

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y : {numpy array},shape (n_samples,)
        input class labels
    n_selected_features : {int}
        number of features to select

    Output
    ------
    X_new: {numpy array},shape (n_samples, n_selected_features)
        data on selected features
    """
    X_new = SelectKBest(f_classif, k=n_selected_features).fit_transform(X, y)
    return X_new