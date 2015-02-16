import scipy.io
import numpy as np


def alpha_investing(X, y, w0, dw):
    y = y - np.mean(y)
    print y


def main():
    # load matlab data
    mat = scipy.io.loadmat('../../data/housingexp.mat')
    X = mat['X']  # data
    y = mat['y']  # label
    X = X.astype(float)
    y = y.astype(float)
    n_samples, n_features = X.shape
    print y.shape
    alpha_investing(X, y, 0.05, 0.05)

if __name__ == '__main__':
    main()
