import scipy.io
from FS_package.function.streaming import alpha_investing


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    X = mat['fea']    # data
    y = mat['gnd']    # label
    y = y[:, 0]
    X = X.astype(float)
    y = y.astype(float)
    sel = alpha_investing.alpha_investing(X, y, 0.05, 0.05)
    print sel

if __name__ == '__main__':
    main()