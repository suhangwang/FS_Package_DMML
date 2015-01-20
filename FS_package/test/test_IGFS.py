import scipy.io
import FS_package.utility.information_gain as ig
from FS_package.function.information_theoretic_based import IGFS


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/iris.mat')
    y = mat['gnd']    # label
    y = y[:, 0]
    X = mat['fea']    # data
    X = X.astype(float)

    # feature weight learning / feature selection
    F = ig.igfs(X, y)
    idx = IGFS.feature_ranking(F)
    print F
    print idx


if __name__ == '__main__':
    main()
