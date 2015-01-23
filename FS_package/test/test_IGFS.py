import numpy as np
import csv
import FS_package.utility.information_gain as ig
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

    # feature weight learning / feature selection
    F = ig.igfs(X, y)
    idx = IGFS.feature_ranking(F)
    print F
    print idx


if __name__ == '__main__':
    main()
