import numpy as np
from FS_package.function.statistics_based import CFS


def main():
    # load data
    mat = np.loadtxt('../data/test_lung_s3.csv', delimiter=',', skiprows=1, usecols=range(0, 101))
    y = mat[:, 0]  # label
    X = mat[:, 1:101]  # data
    X = X.astype(float)
    print np.shape(X)

    # rank feature
    F = CFS.cfs(X, y)

    print 'F'
    print F


if __name__ == '__main__':
    main()