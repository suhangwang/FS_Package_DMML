import scipy.io
from FS_package.utility import construct_W
from FS_package.utility import unsupervised_evaluation
from FS_package.function.sparse_learning_based import MCFS


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    X = mat['fea']
    X = X.astype(float)
    y = mat['gnd']
    y = y[:, 0]

    # construct W
    kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 0.1}
    W = construct_W.construct_W(X, **kwargs)

    # mcfs feature selection
    num_fea = 100
    W = MCFS.mcfs(X, W=W, n_clusters=20, d=num_fea)
    idx = MCFS.feature_ranking(W)

    # evaluation
    num_fea = 100
    selected_features = X[:, idx[0:num_fea]]
    ari, nmi, acc = unsupervised_evaluation.evaluation(selected_features=selected_features, n_clusters=20, y=y)
    print 'ARI:', ari
    print 'NMI:', nmi
    print 'ACC:', acc

if __name__ == '__main__':
    main()