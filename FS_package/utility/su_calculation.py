import numpy as np
import entropy_estimators as ee
import information_gain as ig


def su_calculation(f1, f2):
    """
    This function implement the function which calculates the symmetrical uncertainty, SU
    Input
    ----------
        f1: {numpy array}, shape (n_samples)
            guaranteed to be a discrete data matrix
        f2 : {numpy array}, shape (n_samples)
            guaranteed to be a discrete data matrix
    Output
    ----------
        su: float,
            su(f1,f2) = 2(IG(f1,f2)/H(f1)+H(f2))
    """
    '''
    entropy_estimators.entropyd(x) is used to estimate the discrete entropy given a list of samples of discrete variable x
    '''

    # calculate information gain of f1 and f2, t1 = IG(f1,f2)
    t1 = ig.information_gain(f1, f2)
    # calculate entropy of f1, t2 = H(f1)
    t2 = ee.entropyd(f1)
    # calculate entropy of f2, t3 = H(f2)
    t3 = ee.entropyd(f2)
    # su(f1,f2) = 2(t1/t2+t3)
    su = 2 * np.true_divide(t1, t2+t3)
    return su
