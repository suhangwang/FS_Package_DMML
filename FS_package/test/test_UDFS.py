import scipy.io
from FS_package.utility.unsupervised_evaluation import evaluation
from FS_package.function.sparse_learning_based import UDFS


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    label = mat['gnd']
    label = label[:, 0]
    X = mat['fea']
    X = X.astype(float)

    # UDFS feature selection
    num_fea = 50
    W = UDFS.udfs(X, max_iter= 50, gamma=0.1, k=5, n_clusters=20, verbose=True)
    idx = UDFS.feature_ranking(W)
    selected_features = X[:, idx[0:num_fea]]

    # evaluation
    ari, nmi, acc = evaluation(selected_features=selected_features, n_clusters=20, y=label)
    print 'ARI:', ari
    print 'NMI:', nmi
    print 'ACC:', acc

if __name__ == '__main__':
    main()