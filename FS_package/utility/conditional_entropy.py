import entropy_estimators as ee


def conditional_entropy(f1, f2):
    """
    This function implements conditional entropy calculation, where ce = H(f1) - I(f1;f2)

    Input
    -----
        f1: {numpy array}, shape (n_samples, 1)
            guaranteed to be a discrete one-dimensional numpy array
        f2: {numpy array}, shape (n_samples, 1)
            guaranteed to be a discrete one-dimensional numpy array

    Output
    ------
        ce: {float}
            ce is conditional entropy of f1 and f2
    """

    '''
    entropy_estimators.entropyd(x) is used to estimate the discrete entropy of discrete variable x
    entropy_estimators.midd(x,y) is used to estimate the mutual information between discrete variable x and y
    '''

    ce = ee.entropyd(f1) - ee.midd(f1, f2)
    return ce