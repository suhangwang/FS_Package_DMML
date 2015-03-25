import numpy as np
import csv
from scipy import special


def chisquared_prob(x2, v):
    """
    This function computes the chi-squared probability function.
    P = chisquared_prob(X2,v) returns P(X2|v), the probability of observing a chi-squared value <= X2 with v degrees of freedom.
    Input
    ----------
        x2: {int}
            threshold number for chi-squared value
        v: {int}
            degrees of freedom
    Output
    ----------
        p: {int}
            the probability f observing a chi-squared value <= X2 with v degrees of freedom.
    """
    return special.gammainc(v/2, float(x2)/2)


def cond_indep_G2(Data, x, y, s):
    """
    This function test if x is independent with y given s using likelihood ratio test G2
    Input
    ----------
        x: {int}
            the index of variable x in Data matrix
        y: {int}
            the index of variable y in Data matrix
        s:{{numpy array},shape (n_features, )}
            the indexes of variables in set S
        Data: {numpy array}, shape (n_features, n_samples)
            Input data, guaranteed to be a discrete numpy array
    Output
    ----------
        ci: {int}
            test result(1 = conditional independency, 0 = no)
        G2 = {float}
            G2 value(-1 if not enough data to perform the test -->CI=0)
    ----------
    """
    Data = np.array(Data)
    Data = np.transpose(Data)
    Data = Data.astype(int)
    alpha = 0.05
    n_samples, n_features = Data.shape
    col_max = Data.max(axis=0)
    s_max = np.array([col_max[i] for i in s])

    if len(s) == 0:
        n_ij = np.zeros((col_max[x], col_max[y]))
        t_ij = np.zeros((col_max[x], col_max[y]))
        df = np.prod(np.array([col_max[i] for i in np.array([x, y])])-1) * np.prod(s_max)
    else:
        tmp = np.zeros(len(s_max))
        tmp[0] = 1
        tmp[1:] = np.cumprod(s_max[0:len(s_max)-1])
        qs = 1 + np.dot(s_max-1, tmp)
        n_ijk = np.zeros((col_max[x], col_max[y], qs))
        t_ijk = np.zeros((col_max[x], col_max[y], qs))
        df = np.prod(np.array([col_max[i] for i in np.array([x, y])])-1)*qs

    if (n_samples < 10*df):
        # not enough data to perform the test
        G2 = -1
        ci = 0
    elif (len(s) == 0):
        for i in range(col_max[x]):
            for j in range(col_max[y]):
                col = Data[:, 0]
                n_ij[i, j] = len(col[(Data[:, x] == i+1) & (Data[:, y] == j+1)])
        col_sum_nij = np.sum(n_ij, axis=0)
        if len(col_sum_nij[col_sum_nij == 0]) != 0:
            temp_nij = np.array([])
            for i in range(len(col_sum_nij)):
                if col_sum_nij[i] != 0:
                    temp_nij = np.append(temp_nij, n_ij[:, i])
            temp_nij = temp_nij.reshape((len(temp_nij)/len(n_ij[:, 0], len(n_ij[:, 0]))))
            n_ij = np.transpose(temp_nij)
        row = np.sum(n_ij, axis=1)
        col = np.sum(n_ij, axis=0)
        for i in range(len(row)):
            for j in range(len(col)):
                t_ij[i,j] = float(row[i] * col[j])/n_samples
        tmp = np.zeros((col_max[x], col_max[y]))
        for i in range(col_max[x]):
            for j in range(col_max[y]):
                if t_ij[i, j] == 0:
                    tmp[i, j] = float('Inf')
                else:
                    tmp[i, j] = n_ij[i, j] / t_ij[i, j]
        tmp[(tmp == float('Inf')) | (tmp == 0)] = 1
        tmp = 2 * n_ij * np.log(tmp)

        G2 = np.sum(tmp)
        alpha2 = 1 - chisquared_prob(G2, df)
        ci = (alpha2 >= alpha)
        return int(ci)

    else:
        for example in range(n_samples):
            i = Data[example, x]
            j = Data[example, y]
            si = Data[example, [element for element in s]]-1
            k = int(1+np.dot(si, tmp))
            n_ijk[i-1, j-1, k-1] = n_ijk[i-1, j-1, k-1] + 1
        n_ik = np.sum(n_ijk, axis=1)
        n_jk = np.sum(n_ijk, axis=0)
        n2 = np.sum(n_jk, axis=0)
        tmp = np.zeros((col_max[x], col_max[y], qs))
        for k in range(int(qs)):
            if n2[k] == 0:
                t_ijk[:, :, k] = 0
            else:
                for i in range(col_max[x]):
                    for j in range(col_max[y]):
                        if n2[k] == 0:
                            t_ijk[i, j, k] = float('Inf')
                        else:
                            t_ijk[i, j, k] = n_ik[i, k]*n_jk[j, k]/n2[k]
                        if t_ijk[i, j, k] == 0:
                            tmp[i, j, k] = float('Inf')
                        else:
                            tmp[i, j, k] = n_ijk[i, j, k] / t_ijk[i, j, k]
        tmp[(tmp == float('Inf')) | (tmp == 0)] = 1
        tmp = 2 * n_ijk * np.log(tmp)
        G2 = np.sum(tmp)
        alpha2 = 1 - chisquared_prob(G2, df)
        ci = (alpha2 >= alpha)
        return int(ci)


def main():
    # n_bins = 3
    # num_columns is number of columns in file
    with open('../data/test.csv', 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            num_columns = len(row)
            break
    # load data
    mat = np.loadtxt('../data/test.csv', delimiter=',', skiprows=0, usecols=range(0, num_columns))
    # y = mat[:, 0]  # label
    # X = mat[:, 1:num_columns]  # data
    X = mat
    X = X.astype(int)
    # X = data_discretization.data_discretization(X, n_bins)
    # Data = np.zeros((n_samples, n_features+1))
    #Data[:, 0] = y
    #Data[:, 1:] = X
    x = 0
    y = 1
    s = []

    ci = cond_indep_G2(x, y, s, X)
    print ci


if __name__ == '__main__':
    main()