import scipy.io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from FS_package.function import reliefF
from FS_package.utility import supervised_evaluation


def main():
    # load data
    mat = scipy.io.loadmat('../data/LUNG.mat')
    y = mat['L']  # label
    y = y[:, 0]
    X = mat['M']      # data
    n_samples, n_features = X.shape
    X = X.astype(float)

    # split data
    n_iter = 20
    ss = supervised_evaluation.select_train_split(n_samples, test_size=0.5, n_iter=n_iter)

    # cross validation
    num_fea = 100
    neigh = KNeighborsClassifier(n_neighbors=1)
    correct = 0
    for train, test in ss:
        score = reliefF.feature_select(X[train], y[train])
        idx = reliefF.feature_ranking(score)
        selected_features = X[:, idx[0:num_fea]]
        neigh.fit(selected_features[train], y[train])
        y_predict = neigh.predict(selected_features[test])
        acc = accuracy_score(y[test], y_predict)
        correct = correct + acc
    print 'ACC', float(correct)/n_iter

if __name__ == '__main__':
    main()