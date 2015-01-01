__author__ = 'kewei'
import entropy_estimators as ee


def conditional_entropy(f1, f2):
    """
    This function implement conditional entropy calculation
    Input
    ----------
        f1: {numpy array}, shape (n_samples, 1)
            guaranteed to be a discrete data matrix
        f2: {numpy array}, shape (n_samples, 1)
            guaranteed to be a discrete data matrix
    Output
    ----------
        ce: float
            ce is conditional entropy of f1 and f2
            ce = H(f1) - I(f1;f2)
    ----------
    """
    '''
    entropy_estimators.entropyd(x) is used to estimate the discrete entropy given a list of samples of discrete variable x
    entropy_estimators.midd(x,y) is used to estimate the mutual information between discrete variable x and y
    '''
    ce = ee.entropyd(f1) - ee.midd(f1, f2)
    return ce