__author__ = 'kewei'
import scipy.io
import numpy as np
import entropy_estimators as ee
import conditional_entropy as ce

def information_gain(f1, f2):
    """
    This function implement the function which calculates the information gain
    Input
    ----------
    :param f1:
    :param f2:
    f1: {numpy array}, shape (n_samples)
        a vector
    f2 : {numpy array}, shape (n_samples)
        a vector of the same length as f1
    Output
    ----------
    ig: float,
        ig(f1,f2) = H(f1)-H(f1|f2)
    ----------
    """
    '''
    ee.entropyd(x) is used to estimate the discrete entropy given a list of samples of discrete variable x
    '''
    # IG(f1,f2) = H(f1)-H(f1|f2)
    ig = ee.entropyd(f1) - ce.conditional_entropy(f1, f2)
    return ig