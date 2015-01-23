import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from FS_package.utility.supervised_evaluation import select_train_leave_one_out
from FS_package.function.information_theoretic_based import IGFS


def main():
    # num_columns is number of columns in file
    with open('../data/test_lung_s3.csv', 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            num_columns = len(row)
            break

    # load data
    mat = np.loadtxt('../data/test_lung_s3.csv', delimiter=',', skiprows=1, usecols=range(0, num_columns))
    y = mat[:, 0]  # label
    X = mat[:, 1:num_columns]  # data
    X = X.astype(float)
    n_samples, n_features = X.shape

    loo = select_train_leave_one_out(n_samples)
    neigh = KNeighborsClassifier(n_neighbors=1)

    num_features = 20
    correct = 0
    for train, test in loo:
            # select features
            F = IGFS.igfs(X[train], y[train])
            idx = IGFS.feature_ranking(F)
            features = X[:, idx[0:num_features]]
            neigh.fit(features[train], y[train])
            y_predict = neigh.predict(features[test])
            acc = accuracy_score(y[test], y_predict)
            correct = correct + acc
    print 'LOO error rate', float(1 - (correct/n_samples))


if __name__ == '__main__':
    main()
