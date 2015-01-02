from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation


def evaluation_split(selected_features, y):
    """
    Calculate ACC of the selected features using 50-50 training-test split
    Input
    ----------
        selectedFeatures: {numpy array}, shape (n_samples, n_selectedFeatures}
            data of the selectedFeatures
        Y: {numpy array}, shape (n_samples, 1)
            true labels, guaranteed to be a numpy array
    Output
    ----------
        classification accuracy: {float}
    """

    n_samples, n_features = selected_features.shape

    # repeat 20 times, 50% for training and the rest 50% for testing (default)
    ss = cross_validation.ShuffleSplit(n_samples, n_iter=20, test_size=0.5)

    # 1-nearest neighbor (default)
    neigh = KNeighborsClassifier(n_neighbors=1)

    correct = 0
    for train, test in ss:
        neigh.fit(selected_features[train], y[train])
        y_predict = neigh.predict(selected_features[test])
        acc = accuracy_score(y[test], y_predict)
        correct = correct + acc

    return float(correct)/20


def evaluation_leave_one(selected_features, y):
    """
    Calculate ACC of the selected features using leave-one-out cross validation
    Input
    ----------
        selectedFeatures: {numpy array}, shape (n_samples, n_selectedFeatures}
            data of the selectedFeatures
        Y: {numpy array}, shape (n_samples, 1)
            actual labels
    Output
    ----------
        classification accuracy: {float}
    """

    n_samples, n_features = selected_features.shape
    lo = cross_validation.LeaveOneOut(n_samples)

    # 1-nearest neighbor (default)
    neigh = KNeighborsClassifier(n_neighbors=1)

    correct = 0
    for train, test in lo:
        neigh.fit(selected_features[train], y[train])
        y_predict = neigh.predict(selected_features[test])
        acc = accuracy_score(y[test], y_predict)
        correct = correct + acc

    return float(correct)/n_samples

