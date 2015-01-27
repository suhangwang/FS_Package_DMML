import numpy as np
from FS_package.function.information_theoretic_based import FCBF


def main():
    # num_columns is number of columns in file
    with open('../data/lc.data', 'r') as f:
        for line in f:
            num_columns = len(line.split(','))
            break
    # load data
    mat = np.loadtxt('../data/lc.data', delimiter=',', skiprows=0, usecols=range(0, num_columns))
    X = mat[:, 0:num_columns-1]  # data
    X = X.astype(float)
    y = mat[:, num_columns-1]  # label

    # rank feature
    F = FCBF.fcbf(X, y, delta=0.13)

    # evaluation
    print 'F', F


if __name__ == '__main__':
    main()