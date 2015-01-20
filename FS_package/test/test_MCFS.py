import scipy.io
from FS_package.utility.construct_W import construct_W
from FS_package.utility.unsupervised_evaluation import evaluation
from FS_package.function.sparse_learning_based import MCFS


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    label = mat['gnd']
    label = label[:, 0]
    X = mat['fea']
    X = X.astype(float)

    # construct W
    kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 0.1}
    W = construct_W(X, **kwargs)

    # mcfs feature selection
    num_fea = 200
    selected_features = MCFS.mcfs(X=X, W=W, n_clusters=20, d=num_fea)

    # evaluation
    ari, nmi, acc = evaluation(selected_features=selected_features, n_clusters=20, y=label)
    print 'ARI:', ari
    print 'NMI:', nmi
    print 'ACC:', acc

if __name__ == '__main__':
    main()