__author__ = 'swang187'

import scipy.io
import numpy as np
import scipy.linalg as LA
import utility.sparse_learning as SL
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import accuracy_score


def calculate_obj(X, Y, W, gamma):
    """
    This function calculate the objective function of erfs
    """
    temp = np.dot(X, W) - Y
    return SL.calculate_l21_norm(temp) + gamma*SL.calculate_l21_norm(W)


def erfs(X, Y, **kwargs):
    """
    This function implement efficient and robust feature selection via joint l21-norms minimization
    Input
    --------------
        X: {numpy array}, shape (n_samples, n_features)
            Input data, guaranteed to be a numpy array
        Y: {numpy array}, shape (n_samples, n_classes)
            Each row is a one-hot-coding class label
            guaranteed to be a numpy array
        kwargs: {dictionary}
            gamma: positive scalar
            verbose: boolean
                True if want to display the objective function value
    --------------
    Output:
    --------------
        ind: {numpy array}
    """
    # input error checking
    if 'gamma' not in kwargs:
        print('error, please specify gamma')
        raise
    else:
        gamma = kwargs['gamma']
    if 'max_iter' not in kwargs:
        max_iter = 100
    else:
        max_iter = kwargs['max_iter']
    if 'verbose' not in kwargs:
        verbose = False
    else:
        verbose = kwargs['verbose']

    n_sample, n_feature = X.shape
    A = np.zeros((n_sample, n_sample + n_feature))
    A[:, 0:n_feature] = X
    A[:, n_feature:n_feature+n_sample] = gamma*np.eye(n_sample)
    D = np.eye(n_feature+n_sample)
    for i in range(max_iter):
        # update U as U = D^{-1} A^T (A D^-1 A^T)^-1 Y
        D_inv = LA.inv(D)
        temp = LA.inv(np.dot(np.dot(A, D_inv), A.T))  # (A D^-1 A^T)^-1
        U = np.dot(np.dot(np.dot(D_inv, A.T), temp), Y)
        # update D as D_ii = 1 / 2 / ||U(i,:)||
        D = SL.generate_diagonal_matrix(U)

        # display
        if verbose:
            obj = calculate_obj(X, Y, U[0:n_feature, :], gamma)
            print('obj at iter ' + str(i+1) + ': ' + str(obj) + '\n')

    # the first d rows of U are the feature weights
    feature_weights = U[0:n_feature, :]
    ind = SL.feature_ranking(feature_weights)
    return ind


def main():
    # load MATLAB data
    mat = scipy.io.loadmat('../data/ALLAML.mat')
    label = mat['L']    # label
    label = label[:, 0]
    X = mat['M']    # data
    n_sample, n_feature = X.shape
    X = X.astype(float)
    Y = SL.construct_label_matrix(label)
    # feature weight learning / feature selection
    #idx = erfs(X=X, Y=Y, gamma=0.1, max_iter=50, verbose=True)

    # evalaution
    num_fea = 20
    ss = cross_validation.ShuffleSplit(n_sample, n_iter=5, test_size=0.2)
    clf = svm.LinearSVC()
    mean_acc = 0
    for train, test in ss:
        idx = erfs(X=X[train, :], Y=Y[train, :], gamma=0.1, max_iter=50, verbose=True)
        selected_features = X[:, idx[0:num_fea]]
        clf.fit(selected_features[train, :], label[train])
        y_predict = clf.predict(selected_features[test, :])
        acc = accuracy_score(label[test], y_predict)
        print acc
        mean_acc = mean_acc + acc
    mean_acc = mean_acc / 5
    print mean_acc


if __name__ == '__main__':
    main()

