import entropy_estimators as ee


def information_gain(f1, f2):
    """
    This function implements the function which calculates the information gain, where ig(f1,f2) = H(f1) - H(f1|f2)

    Input
    -----
        f1: {numpy array}, shape (n_samples,)
            guaranteed to be a discrete data array
        f2: {numpy array}, shape (n_samples,)
            guaranteed to be a discrete data array
    Output
    ------
        ig: {float}
    """

    ig = ee.entropyd(f1) - conditional_entropy(f1, f2)
    return ig


def conditional_entropy(f1, f2):
    """
    This function implements conditional entropy calculation, where ce = H(f1) - I(f1;f2)

    Input
    -----
        f1: {numpy array}, shape (n_samples,)
            guaranteed to be a discrete one-dimensional numpy array
        f2: {numpy array}, shape (n_samples,)
            guaranteed to be a discrete one-dimensional numpy array

    Output
    ------
        ce: {float}
            ce is conditional entropy of f1 and f2
    """

    '''
    entropyd(x) is used to estimate the discrete entropy of discrete variable x
    midd(x,y) is used to estimate the mutual information between discrete variable x and y
    '''

    ce = ee.entropyd(f1) - ee.midd(f1, f2)
    return ce