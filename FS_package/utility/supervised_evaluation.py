from sklearn import cross_validation


def select_train_split(n_samples, train_size, n_iters):
    """
    This function implements the function splitting data into training set and test set evenly
    Input
    ----------
    n_samples: {int}
        number of samples in dataset X
    n_iter : {int}
        parameter for the function cross_validation.ShuffleSplit
    test_size : {int}
        percentage of training data, ranging from 0 to 1
    Output
    ----------
    result after splitting the input data X
        Output.train contains the index of samples for training
        Output.test contains the index of samples for testing
    """
    ss = cross_validation.ShuffleSplit(n_samples, n_iter=n_iters, test_size=train_size)
    return ss


def select_train_leave_one_out(n_samples):
    """
    This function implements the leave out out method data splitting
    Input
    ----------
    n_samples: {int}
        number of samples in dataset X
    Output
    ----------
    result after splitting the input data
        Output.train contains the index of features for training
        Output.test contains the index of features for testing
    """
    loo = cross_validation.LeaveOneOut(n_samples)
    return loo



