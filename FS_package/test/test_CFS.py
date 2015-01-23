import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from FS_package.function.statistics_based import CFS


def main():
    # load data
    mat = np.loadtxt('../data/test_lung_s3.csv', delimiter=',', skiprows=1, usecols=range(0, 101))
    X = mat[:, 1:101]  # data
    X = X.astype(float)
    y = mat[:, 0]  # label
    n_samples, n_features = X.shape

    # evalaution
    num_fea = 20
    ss = cross_validation.ShuffleSplit(n_samples, n_iter=5, test_size=0.2)
    clf = svm.LinearSVC()
    mean_acc = 0

    for train, test in ss:
        idx = CFS.cfs(X[train], y[train])
        selected_features = X[:, idx[0:num_fea]]
        clf.fit(selected_features[train], y[train])
        y_predict = clf.predict(selected_features[test])
        acc = accuracy_score(y[test], y_predict)
        print acc
        mean_acc = mean_acc + acc
    mean_acc /= 5
    print mean_acc


if __name__ == '__main__':
    main()