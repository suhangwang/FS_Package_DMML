import scipy.io
from FS_package.function.streaming import alpha_investing


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    X = mat['X']    # data
    y = mat['Y']    # label
    y = y[:, 0]
    X = X.astype(float)
    y = y.astype(float)
    idx = alpha_investing.alpha_investing(X, y, 0.05, 0.05)
    print idx

if __name__ == '__main__':
    main()