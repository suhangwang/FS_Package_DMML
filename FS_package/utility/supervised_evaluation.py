import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation


def evaluation_split(selectedFeatures,Y):
    """
    Calculate ACC of the selected features using 50-50 training-test split
    Input
    ----------
        selectedFeatures: {numpy array}, shape (n_samples, n_selectedFeatures}
            data of the selectedFeatures
        Y: {numpy array}, shape (n_samples, 1)
            actual labels
    """
    N,d = selectedFeatures.shape
    ss = cross_validation.ShuffleSplit(N,n_iter = 20, test_size = 0.5)
    neigh = KNeighborsClassifier(n_neighbors=1)
    correct = 0
    for train, test in ss:
        neigh.fit(selectedFeatures[train],Y[train])
        yPredict = neigh.predict(selectedFeatures[test])
        acc = accuracy_score(Y[test], yPredict)
        correct = correct + acc
        print acc
    return float(correct)/20

def evaluation_leaveOneLabel(selectedFeatures,Y):
    """
    Calculate ACC of the selected features using leave-one-out cross validation
    Input
    ----------
        selectedFeatures: {numpy array}, shape (n_samples, n_selectedFeatures}
            data of the selectedFeatures
        Y: {numpy array}, shape (n_samples, 1)
            actual labels
    """
    N,d = selectedFeatures.shape
    loo = cross_validation.LeaveOneOut(N)
    neigh = KNeighborsClassifier(n_neighbors=1)
    correct = 0
    for train, test in loo:
        neigh.fit(selectedFeatures[train],Y[train])
        yPredict = neigh.predict(selectedFeatures[test])
        acc = accuracy_score(Y[test], yPredict)
        correct = correct + acc
    return float(correct)/N

