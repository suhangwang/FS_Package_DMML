import scipy.io
from FS_package.utility import supervised_evaluation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from FS_package.utility.sparse_learning import *
from FS_package.function.sparse_learning_based import ls_l21_proximal


def main():
    # load data
    mat = scipy.io.loadmat('../data/ORL.mat')
    X = mat['fea']    # data
    X = X.astype(float)
    y = mat['gnd']    # label
    y = y[:, 0]
    n_samples, n_features = X.shape

    # split data
    n_iter = 1
    test_size = 0.5
    ss = supervised_evaluation.select_train_split(n_samples, test_size, n_iter)

    # evaluation
    num_fea = 50
    neigh = KNeighborsClassifier(n_neighbors=1)
    correct = 0

    for train, test in ss:
        Y = construct_label_matrix_pan(y[train])
        W, obj, value_gamma = ls_l21_proximal.proximal_gradient_descent_fast(X[train], Y, 0.1, verbose=False)
        idx = feature_ranking(W)
        selected_features = X[:, idx[0:num_fea]]
        neigh.fit(selected_features[train], y[train])
        y_predict = neigh.predict(selected_features[test])
        acc = accuracy_score(y[test], y_predict)
        correct = correct + acc
    print 'ACC', float(correct)/n_iter


if __name__ == '__main__':
    main()