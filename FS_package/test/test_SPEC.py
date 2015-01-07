import scipy.io
from FS_package.utility import construct_W
from FS_package.function import SPEC
from FS_package.utility import unsupervised_evaluation


def main():
    # load data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    y = mat['gnd']
    y = y[:, 0]
    X = mat['fea']
    X = X.astype(float)

    # build affinity matrix
    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
    W = construct_W.construct_W(X, **kwargs_W)

    # feature selection
    kwargs = {'style': 0, 'W': W}
    score = SPEC.feature_select(X, **kwargs)
    idx = SPEC.feature_ranking(score, **kwargs)

    # evaluation
    num_fea = 100
    selected_features = X[:, idx[0:num_fea]]
    ari, nmi, acc = unsupervised_evaluation.evaluation(selected_features=selected_features, n_clusters=20, y=y)
    print 'ARI:', ari
    print 'NMI:', nmi
    print 'ACC:', acc

if __name__ == '__main__':
    main()