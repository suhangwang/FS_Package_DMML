import scipy.io
from FS_package.utility import construct_W
from FS_package.utility import unsupervised_evaluation
from FS_package.function.sparse_learning_based import MCFS


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    X = mat['X']
    X = X.astype(float)
    y = mat['Y']
    y = y[:, 0]

    # construct W
    kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 0.1}
    W = construct_W.construct_W(X, **kwargs)

    # mcfs feature selection
    n_selected_features = 100
    S = MCFS.mcfs(X, n_selected_features, W=W, n_clusters=20)
    idx = MCFS.feature_ranking(S)

    # evaluation
    num_fea = 100
    selected_features = X[:, idx[0:num_fea]]
    ari, nmi, acc = unsupervised_evaluation.evaluation(selected_features=selected_features, n_clusters=20, y=y)
    print 'ARI:', ari
    print 'NMI:', nmi
    print 'ACC:', acc

if __name__ == '__main__':
    main()