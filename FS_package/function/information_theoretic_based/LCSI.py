from ...utility.entropy_estimators import *


def lcsi(X, y, **kwargs):
    """
    This function implements the basic function of scoring criteria for linear combination of shannon information term
    The scoring criteria is calculated based on the formula j_cmi = I(f;y) - beta * sum(I(fj;f)) + gamma * sum(I(fj;f|y))
    Input
    ----------
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete data matrix
    y : {numpy array}, shape (n_samples, )
        guaranteed to be a numpy array
    kwargs: {dictionary}
        Parameters for different feature selection algorithms.
            beta: {float}
                beta is a parameter in j_cmi = I(f;y) - beta * sum(I(fj;f)) + gamma * sum(I(fj;f|y))
            gamma: {float}
                gamma is a parameter in j_cmi = I(f;y) - beta * sum(I(fj;f)) + gamma * sum(I(fj;f|y))
            function_name: {string}
                indicates which feature selection algorithm we used
            n_selected_features: {int}
                indicates the number of features to select
    Output
    ----------
    F: {numpy array}, shape
        Index of selected features, F(1) is the most important feature.
    """
    n_samples, n_features = X.shape
    # F contains the indexes of selected features, F(1) is the most important feature
    F = []
    # is_n_selected_features_specified indicates that whether user specifies how many features to select
    is_n_selected_features_specified = False
    # initialize the parameters
    if 'beta' in kwargs.keys():
        beta = kwargs['beta']
    if 'gamma' in kwargs.keys():
        gamma = kwargs['gamma']
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_n_selected_features_specified = True
    # For r-th feature we define fr = x[:,r], ,put the unselected fr which has the largest j_cmi into the F
    '''
    midd(x,y) is used to estimate the mutual information between discrete variable x and y
    cmidd(x,y,z) is used to estimated the conditional mutual information between discrete variables
    x and y conditioned on discrete variable z
    '''
    # t1 is a I(f;y) vector for each feature f in X
    t1 = np.zeros(n_features)
    # t2 is a sum(I(fj;f)) vector for each feature f in X
    t2 = np.zeros(n_features)
    # t3 is a sum(I(fj;f|y)) vector for each feature f in X
    t3 = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)
    while True:
        '''
        we define j_cmi as the largest j_cmi of all features
        we define idx as the index of the feature whose j_cmi is the largest
        j_cmi = I(f;y) - beta * sum(I(fj;f)) + gamma * sum(I(fj;f|y))
        '''
        if len(F) == 0:
            # select the feature whose mutual information is the largest as the first
            idx = np.argmax(t1)
            # put the index of feature whose mutual information is the largest into F
            F.append(idx)
            # f_select is the feature we select
            f_select = X[:, idx]

        if is_n_selected_features_specified is True:
            if len(F) == n_selected_features:
                break
        if is_n_selected_features_specified is not True:
            if j_cmi <= 0:
                break

        # we assign an extreme small value to j_cmi in order to make sure it is smaller then possible value of j_cmi
        j_cmi = -1000000000000
        if 'function_name' in kwargs.keys():
            if kwargs['function_name'] == 'MRMR':
                beta = 1.0 / len(F)
            elif kwargs['function_name'] == 'JMI':
                beta = 1.0 / len(F)
                gamma = 1.0 / len(F)
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t2[i] += midd(f_select, f)
                t3[i] += cmidd(f_select, f, y)
                # calculate j_cmi for feature i (not in F)
                t = t1[i] - beta * t2[i] + gamma * t3[i]
                # record the largest j_cmi and its idx
                if t > j_cmi:
                    j_cmi = t
                    idx = i
        # put the index of feature whose j_cmi is the largest into F
        F.append(idx)
        # f_select is the feature we select
        f_select = X[:, idx]
    return np.array(F)





