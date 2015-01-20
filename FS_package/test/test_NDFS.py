import scipy.io
import numpy as np
from FS_package.utility import construct_W
from FS_package.function.sparse_learning_based import NDFS
from FS_package.utility import unsupervised_evaluation


def main():
    # load data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    y = mat['gnd']    # label
    y = y[:, 0]
    X = mat['fea']    # data
    X = X.astype(float)

    kwargs = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
    W = construct_W.construct_W(X, **kwargs)

    ######################################
    # needs to be fixed
    L = np.diag(W.sum(1)) - W
    ######################################

    # feature weight learning / feature selection
    W, obj = NDFS.ndfs(X, n_clusters=40, L=L, verbose=1, max_iter=30)
    idx = NDFS.feature_ranking(W)

    # evaluation
    num_fea = 100
    selected_features = X[:, idx[0:num_fea]]
    ari, nmi, acc = unsupervised_evaluation.evaluation(selected_features=selected_features, n_clusters=20, y=y)
    print 'ARI:', ari
    print 'NMI:', nmi
    print 'ACC:', acc

if __name__ == '__main__':
    main()
