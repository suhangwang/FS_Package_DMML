import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from FS_package.utility import supervised_evaluation
from FS_package.function.information_theoretic_based import JMI


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
    # split data into train and test sets
    loo = supervised_evaluation.select_train_leave_one_out(n_samples)
    neigh = KNeighborsClassifier(n_neighbors=3)
    num_fea = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

    for i in range(len(num_fea)):
        correct = 0
        j = 0
        for train, test in loo:
            # select features
            F = JMI.jmi(X[train], y[train], n_selected_features=num_fea[i])
            features = X[:, F]
            neigh.fit(features[train], y[train])
            y_predict = neigh.predict(features[test])
            acc = accuracy_score(y[test], y_predict)
            correct = correct + acc
            j += 1
        print 'LOO error rate', num_fea[i], float(1 - (correct/j))


if __name__ == '__main__':
    main()


