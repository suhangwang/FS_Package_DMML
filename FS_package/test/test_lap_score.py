import scipy.io
from FS_package.function import lap_score
from FS_package.utility import construct_W
from FS_package.utility import unsupervised_evaluation


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/ORL.mat')
    y = mat['gnd']    # label
    y = y[:, 0]
    X = mat['fea']    # data
    n_samples, n_features = X.shape
    X = X.astype(float)

    # construct affinity matrix
    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
    W = construct_W.construct_W(X, **kwargs_W)

    # feature selection
    score = lap_score.feature_select(X, W = W)
    idx = lap_score.feature_ranking(score)

    # evaluation
    num_fea = 100
    selected_features = X[:, idx[0:num_fea]]
    ari, nmi, acc = unsupervised_evaluation.evaluation(selected_features=selected_features, n_clusters=40, y=y)
    print 'ARI:', ari
    print 'NMI:', nmi
    print 'ACC:', acc

if __name__ == '__main__':
    main()