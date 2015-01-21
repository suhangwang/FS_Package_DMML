import scipy.io
from FS_package.utility.sparse_learning import *
from FS_package.function.sparse_learning_based import ls_l21_proximal


def main():
    # load data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    X = mat['fea']    # data
    X = X.astype(float)
    label = mat['gnd']    # label
    label = label[:, 0]
    Y = construct_label_matrix(label)
    ls_l21_proximal.proximal_gradient_descent(X, Y, 0.1)


if __name__ == '__main__':
    main()