__author__ = 'kewei'

import scipy.io
import numpy as np
import utility.entropy_estimators as ee
import utility.information_gain as ig
import utility.conditional_entropy as ce
from utility.supervised_evaluation import evaluation_leaveOneLabel

def lcsi(X, y, **kwargs):
    """
    This function implement the a scoring criteria for linear combination of shannon information term
    Input
    ----------
    :param X:
    :param y:
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete data matrix
    y : {numpy array}, shape (n_samples, label)
        guaranteed to be a numpy array
    kwargs: {dictionary}
        Parameters for different feature selection algorithms.
            beta: int
                beta is a parameter depend on the feature selection algorithm
            gamma: int
                gamma is a parameter depend on the feature selection algorithm
            functionName: {string}
                Indicates which feature selection algorithm we used
    Output
    ----------
    F: {numpy array}, shape
        a list containing the ranked features,F(1) is the most important feature.
    ----------
    """
    n_samples, n_features = X.shape
    # F is the matrix for index of ranked feature,F(1) is the most important feature.
    F = []
    # initialize the parameters
    if 'beta' in kwargs.keys():
        beta = kwargs['beta']
    if 'gamma' in kwargs.keys():
        gamma = kwargs['gamma']
    # For r-th feature we define fr = x[:,i] ,put the unselected fr which has the largest j_cmi into the F
    '''
    ee.midd(x,y) is used to estimate the mutual information between discrete variable x and y
    ee.cmidd(x,y,z) is used to estimated the conditional mutual information between discrete variables x and y conditioned on discrete variable z
    '''
    while len(F) < n_samples:
        # we define j_cmi as the maximum j_cmi of all features
        # we define idx as the index of the feature whose j_cmi is the max
        # j_cmi = I(f;y) - beta * sum(I(fj;f)) + gamma * sum(I(fj;f|y))
        # initialize j_cmi and idx
        j_cmi = -1000000000000
        idx = -1
        if 'functionName' in kwargs.keys():
            if kwargs['functionName'] == 'JMI':
                if len(F) != 0:
                    beta = np.true_divide(1, len(F))
                else:
                    beta = np.true_divide(1, 1)
            elif kwargs['functionName'] == 'MRMR':
                if len(F) != 0:
                    beta = np.true_divide(1, len(F))
                    gamma = np.true_divide(1, len(F))
                else:
                    beta = np.true_divide(1, 1)
                    gamma = np.true_divide(1, 1)
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                # t1 =I(f;y)
                t1 = ee.midd(f, y)
                # t2 = sum(I(fj;f)), t3 = sum(I(fj;f|y))
                # initialize t2,t3
                t2 = 0
                t3 = 0
                for j in F:
                    if j != i:
                        fj = X[:, j]
                        t2 += ee.midd(fj, f)
                        t3 += ee.cmidd(fj, f, y)
                # calculate j_cmi for feature i, we define it as t
                t = t1 - beta * t2 + gamma * t3
                # store the biggest j_cmi and its idx
                if t > j_cmi:
                    j_cmi = t
                    idx = i
        # put the idx of feature whose j_cmi is the max into F
        F.append(idx)

    return F

def mifs(X, y, **kwargs):
    """
    This function implement the mifs function
    Input
    ----------
    :param X:
    :param y:
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete data matrix
    y : {numpy array}, shape (n_samples, label)
        guaranteed to be a numpy array
    kwargs: {dictionary}
        Parameters for different feature selection algorithms.
            beta: int
                beta is a parameter depend on the feature selection algorithm
            gamma: int
                gamma is a parameter depend on the feature selection algorithm
            functionName: {string}
                Indicates which feature selection algorithm we used
    Output
    ----------
    F: {numpy array}, shape
        a list containing the ranked features,F(1) is the most important feature.
    ----------
    """
    if 'beta' not in kwargs.keys():
        beta = 0.5
    else:
        beta = kwargs['beta']
    F = lcsi(X, y, beta=beta, gamma=0)
    return F

def mim(X, y, **kwargs):
    """
    This function implement the mim function
    Input
    ----------
    :param X:
    :param y:
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete data matrix
    y : {numpy array}, shape (n_samples, label)
        guaranteed to be a numpy array
    kwargs: {dictionary}
        Parameters for different feature selection algorithms.
            beta: int
                beta is a parameter depend on the feature selection algorithm
            gamma: int
                gamma is a parameter depend on the feature selection algorithm
            functionName: {string}
                Indicates which feature selection algorithm we used
    Output
    ----------
    F: {numpy array}
        a list containing the ranked features,F(1) is the most important feature.
    ----------
    """
    F = lcsi(X, y, beta=0, gamma=0)
    return F

def cife(X, y, **kwargs):
    """
    This function implement the cife function
    Input
    ----------
    :param X:
    :param y:
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete data matrix
    y : {numpy array}, shape (n_samples, label)
        guaranteed to be a numpy array
    kwargs: {dictionary}
        Parameters for different feature selection algorithms.
            beta: int
                beta is a parameter depend on the feature selection algorithm
            gamma: int
                gamma is a parameter depend on the feature selection algorithm
            functionName: {string}
                Indicates which feature selection algorithm we used
    Output
    ----------
    F: {numpy array}, shape
        a list containing the ranked features,F(1) is the most important feature.
    ----------
    """
    F = lcsi(X, y, beta=1, gamma=1)
    return F

def jmi(X, y, **kwargs):
    """
    This function implement the jmi function
    Input
    ----------
    :param X:
    :param y:
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete data matrix
    y : {numpy array}, shape (n_samples, label)
        guaranteed to be a numpy array
    kwargs: {dictionary}
        Parameters for different feature selection algorithms.
            beta: int
                beta is a parameter depend on the feature selection algorithm
            gamma: int
                gamma is a parameter depend on the feature selection algorithm
            functionName: {string}
                Indicates which feature selection algorithm we used
    Output
    ----------
    F: {numpy array}, shape
        a list containing the ranked features,F(1) is the most important feature.
    ----------
    """
    F = lcsi(X, y, gamma=0, functionName='JMI')
    return F

def mrmr(X, y, **kwargs):
    """
    This function implement the mrmr function
    Input
    ----------
    :param X:
    :param y:
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete data matrix
    y : {numpy array}, shape (n_samples, label)
        guaranteed to be a numpy array
    kwargs: {dictionary}
        Parameters for different feature selection algorithms.
            beta: int
                beta is a parameter depend on the feature selection algorithm
            gamma: int
                gamma is a parameter depend on the feature selection algorithm
            functionName: {string}
                Indicates which feature selection algorithm we used
    Output
    ----------
    F: {numpy array}, shape
        a list containing the ranked features,F(1) is the most important feature.
    ----------
    """
    F = lcsi(X, y, functionName='MRMR')
    return F

def cmim(X, y, **kwargs):
    """
    This function implement the a method of a scoring criteria as non-linear combination of shannon information term,CMIM
    Input
    ----------
    :param X:
    :param y:
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete data matrix
    y : {numpy array}, shape (n_samples, label)
        guaranteed to be a numpy array
    kwargs: {dictionary}
        Parameters for extend purpose
    Output
    ----------
    F: {numpy array}, shape
        a list containing the ranked features,F(1) is the most important feature.
    ----------
    """
    n_samples, n_features = X.shape
    # F is the matrix for index of ranked feature,F(1) is the most important feature.
    F = []

    # For r-th feature we define fr = x[:,i] ,put the unselected fr which has the largest j_cmim into the F
    '''
    ee.midd(x,y) is used to estimate the mutual information between discrete variable x and y
    ee.cmidd(x,y,z) is used to estimated the conditional mutual information between discrete variables x and y conditioned on discrete variable z
    '''
    while len(F) < n_samples:
        # we define j_cmim as the maximum j_cmim of all features
        # j_cmim = I(f;y)-max(I(f;fj)-I(f;fj|y)),fj is defined as all feature in F
        # we define idx as the index of the feature whose j_cmim is the max
        # initialize j_cmim and idx
        j_cmim = -1000000000000
        idx = -1
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                # t1 =I(f;y)
                t1 = ee.midd(f, y)
                # t2 = I(f;fj), t3 = I(f;fj|y), max is defined as the maximum of t2-t3
                # initialize t2,t3,max
                t2 = 0
                t3 = 0
                max = -10000000
                for j in F:
                    if j != i:
                        fj = X[:, j]
                        t2 = ee.midd(f, fj)
                        t3 = ee.cmidd(f, fj, y)
                        if t2-t3 > max:
                            max = t2-t3
                # calculate j_cmim for feature i, we define it as t
                t = t1 - max
                # store the biggest j_cmim and its idx
                if t > j_cmim:
                    j_cmim = t
                    idx = i
        # put the idx of feature whose j_cmim is the max into F
        F.append(idx)

    return F

def If(X, y, **kwargs):
    """
    This function implement the a method of a scoring criteria as non-linear combination of shannon information term,IF
    Input
    ----------
    :param X:
    :param y:
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete data matrix
    y : {numpy array}, shape (n_samples, label)
        guaranteed to be a numpy array
    kwargs: {dictionary}
        Parameters for extend purpose
    Output
    ----------
    F: {numpy array}, shape
        a list containing the ranked features,F(1) is the most important feature.
    ----------
    """
    F = cmim(X, y)
    return F

def icap(X, y, **kwargs):
    """
    This function implement the a method of a scoring criteria as non-linear combination of shannon information term,ICAP
    Input
    ----------
    :param X:
    :param y:
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete data matrix
    y : {numpy array}, shape (n_samples, label)
        guaranteed to be a numpy array
    kwargs: {dictionary}
        Parameters for extend purpose
    Output
    ----------
    F: {numpy array}, shape
        a list containing the ranked features,F(1) is the most important feature.
    ----------
    """
    n_samples, n_features = X.shape
    # F is the matrix for index of ranked feature,F(1) is the most important feature.
    F = []

    # For r-th feature we define fr = x[:,i] ,put the unselected fr which has the largest j_icap into the F
    '''
    ee.midd(x,y) is used to estimate the mutual information between discrete variable x and y
    ee.cmidd(x,y,z) is used to estimated the conditional mutual information between discrete variables x and y conditioned on discrete variable z
    '''
    while len(F) < n_samples:
        # we define j_icap as the maximum j_icap of all features
        # j_icap = I(f;y)-sum(max(0,(I(f;fj)-I(f;fj|y)))),fj is defined as all feature in F
        # we define idx as the index of the feature whose j_icap is the max
        # initialize j_icap and idx
        j_icap = -1000000000000
        idx = -1
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                # t1 =I(f;y)
                t1 = ee.midd(f, y)
                # t2 = I(f;fj), t3 = I(f;fj|y), sum is defined as the sum of positive t2-t3
                # initialize t2,t3,sum
                t2 = 0
                t3 = 0
                sum = 0
                for j in F:
                    if j != i:
                        fj = X[:, j]
                        t2 = ee.midd(f, fj)
                        t3 = ee.cmidd(f, fj, y)
                        if t2-t3 > 0:
                            sum += t2-t3
                # calculate j_icap for feature i, we define it as t
                t = t1 - sum
                # store the biggest j_icap and its idx
                if t > j_icap:
                    j_icap = t
                    idx = i
        # put the idx of feature whose j_icap is the max into F
        F.append(idx)

    return F

def disr(X, y, **kwargs):
    """
    This function implement the a method of a scoring criteria as non-linear combination of shannon information term,DISR
    Input
    ----------
    :param X:
    :param y:
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete data matrix
    y : {numpy array}, shape (n_samples, label)
        guaranteed to be a numpy array
    kwargs: {dictionary}
        Parameters for extend purpose
    Output
    ----------
    F: {numpy array}, shape
        a list containing the ranked features,F(1) is the most important feature.
    ----------
    """
    n_samples, n_features = X.shape
    # F is the matrix for index of ranked feature,F(1) is the most important feature.
    F = []

    # For r-th feature we define fr = x[:,i] ,put the unselected fr which has the largest j_disr into the F
    '''
    ee.midd(x,y) is used to estimate the mutual information between discrete variable x and y
    ee.cmidd(x,y,z) is used to estimated the conditional mutual information between discrete variables x and y conditioned on discrete variable z
    '''
    while len(F) < n_samples:
        # we define j_disr as the maximum j_disr of all features
        # j_disr = sum(I(fk,j;y)|H(fk,j,y)),fj is defined as all feature in F
        # we define idx as the index of the feature whose j_disr is the max
        # initialize j_disr and idx
        j_disr = -1000000000000
        idx = -1
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                if len(F) == 0:
                    # if F is empty, calculate information gain for feature i, we define it as t
                    t = ig.information_gain(f, y)
                else:
                    # t = sum(I(fk,j;y)|H(fk,j,y)),fj is defined as all feature in F
                    # initialize t
                    t = 0
                    for j in F:
                        if j != i:
                            fj = X[:, j]
                            # t1 =I(fk,j;y) =I(fj;y)+I(f;y|fj)
                            t1 = ee.midd(fj, y) + ee.cmidd(f, y, fj)
                            # t2 = H(f fj y)=H(f) + H(fj|f) + H(y|f,fj)
                            # H(y|f,fj) = H(y|fj)-I(y;f|fj)
                            t2 = ee.entropyd(f) + ce.conditional_entropy(fj, f) + (ce.conditional_entropy(y, fj) - ee.cmidd(y, f, fj))
                            t += np.true_divide(t1, t2)
                # store the biggest j_disr and its idx
                if t > j_disr:
                    j_disr = t
                    idx = i
        # put the idx of feature whose j_disr is the max into F
        F.append(idx)

    return F

def main():
    # load data
    mat = np.loadtxt('../data/test_colon_s3.csv', delimiter=',', skiprows=1, usecols=range(0, 2001))
    y = mat[:, 0]  # label
    X = mat[:, 1:2001]  # data
    X = X.astype(float)

    # rank feature
    F = disr(X, y)

    # evaluation
    numFea = 15
    selectedFeatures = X[:, F[0:numFea]]
    print selectedFeatures
    ACC = evaluation_leaveOneLabel(selectedFeatures=selectedFeatures, Y=y)
    print ACC

if __name__ == '__main__':
    main()




