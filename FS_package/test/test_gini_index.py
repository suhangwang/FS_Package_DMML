import scipy.io
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score
from FS_package.function.statistics_based import gini_index
from sklearn import cross_validation


def main():
    # load data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    y = mat['gnd']
    y = y[:, 0]
    X = mat['fea']
    n_samples, n_features = X.shape
    X = X.astype(float)

    # split data
    n_iter = 20
    test_size = 0.5
    ss = cross_validation.select_train_split(n_samples, test_size, n_iter)

    # cross validation
    num_fea = 100
    clf = svm.LinearSVC()
    correct = 0

    for train, test in ss:
        score = gini_index.gini_index(X[train], y[train])
        idx = gini_index.feature_ranking(score)
        selected_features = X[:, idx[0:num_fea]]
        clf.fit(selected_features[train], y[train])
        y_predict = clf.predict(selected_features[test])
        acc = accuracy_score(y[test], y_predict)
        correct = correct + acc
    print 'ACC', float(correct)/n_iter

if __name__ == '__main__':
    main()

