import scipy.io
import numpy as np


def feature_ranking(W):
    """
    This function implement the gini_index ranking function

    Input
    ----------
    W:{numpy array}, shape (n_samples, gini_index)
      a list containing the gini index of each feature
    ----------
    Output
    ----------
    idx: {numpy array}, shape (n_samples, gini_index)
        a list of features ranking by their ability to classify the data.f_list(1) is
        the most important feature.
    ----------
    """
    # rank from small to big
    idx = np.argsort(W)
    return idx


def gini_index(X, y):
    """
    This function implement the gini_index function

    Input
    ----------
    X: {numpy array}, shape (n_samples, n_features)
        guaranteed to be a numpy array
    y : {numpy array}, shape (n_samples, label)
        guaranteed to be a numpy array
    ----------
    Output
    ----------
    W: {numpy array}, shape (n_samples, gini_index)
        a list containing the gini index of each feature
    ----------
    """
    n_samples, n_features = X.shape
    # w is the matrix for the gini index of each feature
    # initialize
    W = np.ones(n_features) * 0.5
    # For r-th feature we define fr = x[:,i] ,v include all unique values in fr
    for i in range(n_features):
        v = np.unique(X[:, i])
        for j in range(len(v)):
            '''
          For r-th feature, left_y include all the corresponding labels of instance whose r-th feature value less or equal to v[j]
          right_y include all the corresponding labels of instance whose r-th feature value more than v[j]
          '''
            left_y = y[X[:, i] <= v[j]]
            right_y = y[X[:, i] > v[j]]

            # for v[i], gini_left is the sum of square of probability of occurrence of v[i] in left_y
            # for v[i], gini_left is the sum of square of probability of occurrence of v[i] in right_y
            gini_left = 0
            gini_right = 0
            # k is define as the all different labels in y
            for k in range(np.min(y), np.max(y)+1):
                if len(left_y) != 0:
                    # t1_left is define as the probability of occurrence of k in left_y
                    t1_left = np.true_divide(len(left_y[left_y == k]), len(left_y))
                    # t2_left is define as the the square of t1_left
                    t2_left = np.power(t1_left, 2)
                    gini_left += t2_left

                if len(right_y) != 0:
                    # t1_right is define as the probability of occurrence of k in left_y
                    t1_right = np.true_divide(len(right_y[right_y == k]), len(right_y))
                    # t2_right is define as the the square of t1_right
                    t2_right = np.power(t1_right, 2)
                    gini_right += t2_right

            gini_left = 1 - gini_left
            gini_right = 1 - gini_right

            # t1_gini is weighted average of length of left_y and right_y
            t1_gini = (len(left_y) * gini_left + len(right_y) * gini_right)

            # For r-th feature, gini is define as the gini index of it
            gini = np.true_divide(t1_gini, len(y))

            # For r-th feature, gini can not bigger than 0.5
            if gini < W[i]:
                W[i] = gini
    return W


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/iris.mat')
    y = mat['gnd']  # label
    y = y[:, 0]
    X = mat['fea']  # data

    X = X.astype(float)

    # feature weight learning / feature selection
    W = gini_index(X, y)
    print W

    idx = feature_ranking(W)

    # evaluation
    ranked_features = X[:, idx]
    print ranked_features

if __name__ == '__main__':
    main()








