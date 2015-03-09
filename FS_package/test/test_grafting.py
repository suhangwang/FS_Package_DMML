import scipy.io
from FS_package.function.streaming import grafting


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/USPS.mat')
    X = mat['fea']    # data
    y = mat['gnd']    # label
    y = y[:, 0]
    X = X.astype(float)
    y = y.astype(float)
    idx = grafting.grafting(X, y, 0.1)
    print idx

if __name__ == '__main__':
    main()