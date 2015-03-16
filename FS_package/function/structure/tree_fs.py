import math
from ...utility.sparse_learning import *


def group_fs(X, y, z, idx, **kwargs):
    """
    This function implements supervised sparse group feature selection with least square loss, i.e.,
    min_{w} ||Xw-Y||_2^2 + z_1||x||_1 + z_2*sum_j w_j||w_{G_{j}}||
    --------------------------
    Input
        X: {numpy array}, shape (n_samples, n_features)
            input data, guaranteed to be a numpy array
        Y: {numpy array}, shape (n_samples, n_classes)
            each row is a one-hot-coding class label, guaranteed to be a numpy array
        z: {float}
            regularization parameter of L2 norm for the non-overlapping group
        idx: {numpy array}, shape (3, n_nodes)
            3*nodes matrix, where nodes denotes the number of nodes of the tree
            idx(1,:) contains the starting index
            idx(2,:) contains the ending index
            idx(3,:) contains the corresponding weight (w_{j})
        kwargs : {dictionary}
            verbose: {boolean} True or False
                True if user want to print out the objective function value in each iteration, False if not
    --------------------------
    Output
        w: {numpy array}, shape (n_features, )
            weight matrix
        obj: {numpy array}, shape (n_iterations, )
            objective function value during iterations
        value_gamma: {numpy array}, shape (n_iterations, )
            suitable step size during iterations

    Reference:
        Liu, Jun, et al. "Moreau-Yosida Regularization for Grouped Tree Structure Learning." NIPS. 2010.
    """

    if 'verbose' not in kwargs:
        verbose = False
    else:
        verbose = kwargs['verbose']

    # Starting point initialization #
    n_samples, n_features = X.shape
    # compute X'y
    Xty = np.dot(np.transpose(X), y)
    # initialize a starting point
    w = np.zeros(n_features)
    # compute Xw = X*w
    Xw = np.dot(X, w)

    # Starting the main program, the Armijo Goldstein line search scheme + accelerated gradient descent
    # initialize step size gamma = 1
    gamma = 1
    # assign wp with w, and Xwp with Xw
    wp = w
    Xwp = Xw
    wwp = np.zeros(n_features)
    alphap = 0
    alpha = 1
    # indicates whether the gradient step only changes a little
    flag = False

    max_iter = 1000
    value_gamma = np.zeros(max_iter)
    obj = np.zeros(max_iter)
    for iter_step in range(max_iter):
        # step1: compute search point s based on wp and w (with beta)
        beta = (alphap-1)/alpha
        s = w + beta*wwp
        # step2: line search for gamma and compute the new approximation solution w
        Xs = Xw + beta*(Xw - Xwp)
        # compute X'* Xs
        XtXs = np.dot(np.transpose(X), Xs)
        # obtain the gradient g
        G = XtXs - Xty
        # copy w and Xw to wp and Xwp
        wp = w
        Xwp = Xw

        while True:
            # let s walk in a step in the antigradient of s to get v and then do the L1/L2-norm regularized projection
            v = s - G/gamma

            # tree overlapping group lasso projection
            n_nodes = int(idx.shape[1])
            idx[2, :] = idx[2, :] * z / gamma
            w = tree_lasso_projection(v, n_features, idx, n_nodes)
            # the difference between the new approximate solution w and the search point s
            v = w - s
            # compute Xw = X*w
            Xv = Xw - Xs
            r_sum = np.inner(v, v)
            l_sum = np.inner(Xv, Xv)

            # determine weather the gradient step makes little improvement
            if r_sum <= 1e-20:
                flag = True
                break

            # the condition is ||Xv||_2^2 <= gamma * ||v||_2^2
            if l_sum < r_sum*gamma:
                break
            else:
                gamma = max(2*gamma, l_sum/r_sum)
            value_gamma[iter_step] = gamma

            # step3: update alpha and alphap, and check weather converge
            alphap = alpha
            alpha = (1+math.sqrt(4*alpha*alpha+1))/2

            wwp = w - wp
            Xwy = Xw -y
            # calculate the regularization part
            tree_norm_val = tree_norm(w, n_features, idx, n_nodes)

            # function value = loss + regularization
            obj[iter_step] = np.inner(Xwy, Xwy)/2 + z*tree_norm_val

            if verbose:
                print 'obj at iter ' + str(iter_step+1) + ': ' + str(obj[iter_step])

            if flag is True:
                break

            # determine weather converge
            if iter_step >= 2 and math.fabs(obj[iter_step] - obj[iter_step-1]) < 1e-3:
                break
        return w, obj, value_gamma




