import numpy as np
import csv
from FS_package.function.statistics_based import anova_f_value


def main():
    # get num_features
    with open('../data/test_lung_s3.csv', 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            num_columns = len(row)
            break

    # load data
    mat = np.loadtxt('../data/test_lung_s3.csv', delimiter=',', skiprows=1, usecols=range(0, num_columns))
    X = mat[:, 1:num_columns]  # data
    X = X.astype(float)
    y = mat[:, 0]  # label

    # feature selection
    num_fea = 5
    F = anova_f_value.anova_f_value(X, y, num_fea)
    print F


if __name__ == '__main__':
    main()
