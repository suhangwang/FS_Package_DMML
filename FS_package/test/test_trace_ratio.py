import scipy.io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from FS_package.utility import supervised_evaluation
from FS_package.function.similarity_based import trace_ratio


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    X = mat['fea']  # data
    y = mat['gnd']  # label
    y = y[:, 0]
    X = X.astype(float)
    n_samples, n_features = X.shape

    # split data
    n_iter = 20
    test_size = 0.5
    ss = supervised_evaluation.select_train_split(n_samples, test_size, n_iter)

    # evaluation
    num_fea = 100
    neigh = KNeighborsClassifier(n_neighbors=1)
    correct = 0

    for train, test in ss:
        idx, feature_score, subset_score = trace_ratio.trace_ratio(X, y, num_fea, style='fisher')
        selected_features = X[:, idx]
        neigh.fit(selected_features[train], y[train])
        y_predict = neigh.predict(selected_features[test])
        acc = accuracy_score(y[test], y_predict)
        correct = correct + acc
    print 'ACC', float(correct)/n_iter


if __name__ == '__main__':
    main()
