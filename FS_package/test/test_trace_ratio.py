import scipy.io
from FS_package.function.similarity_based import trace_ratio


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    label = mat['gnd']
    y = label[:, 0]
    X = mat['fea']
    n_samples, n_features = X.shape
    X = X.astype(float)

    trace_ratio.trace_ratio(X, y, style='laplacian')


if __name__ == '__main__':
    main()
