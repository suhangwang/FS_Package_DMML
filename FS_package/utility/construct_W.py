import numpy as np
from scipy.sparse import *
from sklearn.metrics.pairwise import pairwise_distances


def construct_W(X, **kwargs):
    """ Construct the affinity matrix W
    If kwargs is none, use the default parameter settings
    If kwargs is not null, construct the affinity matrix according to parameters in kwargs
    Input
    ----------
    data: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a numpy array
    kwargs: {dictionary}
        Parameters to construct different affinity matrix W:
            y: {numpy array}, shape (n_samples, 1)
                The parameter needed under the 'supervised' neighbor mode
            metric: {string}
                Choices for different distance measure
                'euclidean' - use euclidean distance
                'cosine' - use cosine distance (default)
            neighbor_mode: {string}
                Indicates how to construct the graph
                'knn' - Put an edge between two nodes if and only if they are among the
                        k nearest neighbors of each other (default)
                'supervised' - Put an edge between two nodes if they belong to same class
                        and they are among the k nearest neighbors of each other
            weight_mode: {string}
                Indicates how to assign weights for each edge in the graph
                'binary' - 0-1 weighting, every edge receives weight of 1 (default)
                'heat_kernel' - If nodes i and j are connected, put weight W_ij = exp(-norm(x_i - x_j)/2t^2)
                                This weight mode can only be used under 'euclidean' metric and you are required
                                to provide the parameter t
                'cosine' - If nodes i and j are connected, put weight cosine(x_i,x_j).
                            Can only be used under 'cosine' metric
            k: {int}
                Choices for the number of neighbors (default k = 5)
            t: {float}
                Parameter for the 'heat_kernel' weight_mode
            fisher_score: {boolean}
                Indicates whether to build the affinity matrix in a fisher_score way, in which W_ij = 1/n_l
                if yi = yj = l;
                otherwise W_ij = 0 (default fisher_score = false)
            relief: {boolean}
                Indicates whether to build the affinity matrix in a relief way, NH(x) or NM(x,y) denotes a set of
                k nearest points to x with the same class of x, or a different class (the class y), respectively.
                W_ij = 1 if i = j; W_ij = -1/k if x_j \in NH(x_i); W_ij = 1/(c-1)k if xj \in NM(x_i, y)
                (default relief = false)
    Output
    -------
    W: {sparse matrix}, shape (n_samples, n_samples)
        Output affinity matrix W, guaranteed to be a numpy array.
    """

    # default metric is 'cosine'
    if 'metric' not in kwargs.keys():
        kwargs['metric'] = 'cosine'

    # default neighbor mode is 'knn' and default neighbor size is 5
    if 'neighbor_mode' not in kwargs.keys():
        kwargs['neighbor_mode'] = 'knn'
    if kwargs['neighbor_mode'] == 'knn' and 'k' not in kwargs.keys():
        kwargs['k'] = 5
    if kwargs['neighbor_mode'] == 'supervised' and 'k' not in kwargs.keys():
        kwargs['k'] = 5
    if kwargs['neighbor_mode'] == 'supervised' and 'y' not in kwargs.keys():
        print ('Warning: label is required in the supervised neighborMode!!!')
        exit(0)

    # default weight mode is 'binary', default t in heat kernel mode is 1
    if 'weight_mode' not in kwargs.keys():
        kwargs['weight_mode'] = 'binary'
    if kwargs['weight_mode'] == 'heat_kernel':
        if kwargs['metric'] != 'euclidean':
            kwargs['metric'] = 'euclidean'
        if 't' not in kwargs.keys():
            kwargs['t'] = 1
    elif kwargs['weight_mode'] == 'cosine':
        if kwargs['metric'] != 'cosine':
            kwargs['metric'] = 'cosine'

    # default fisher_score and relief mode are 'false'
    if 'fisher_score' not in kwargs.keys():
        kwargs['fisher_score'] = False
    if 'relief' not in kwargs.keys():
        kwargs['relief'] = False

    n_samples, n_features = np.shape(X)

    # choose 'knn' neighbor mode
    if kwargs['neighbor_mode'] == 'knn':
        k = kwargs['k']
        if kwargs['weight_mode'] == 'binary':
            if kwargs['metric'] == 'euclidean':
                # compute pairwise euclidean distances
                D = pairwise_distances(X)
                # sort the distance matrix D in ascending order
                dump = np.sort(D, axis=1)
                idx = np.argsort(D, axis=1)
                # choose the k-nearest neighbors for each instance
                idx_new = idx[:, 0:k+1]
                dump_new = dump[:, 0:k+1]
                G = np.zeros((n_samples*(k+1), 3))
                G[:, 0] = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
                G[:, 1] = np.ravel(idx_new, order='F')
                G[:, 2] = np.ravel(dump_new, order='F')
                # build sparse affinity matrix W
                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                W[W > 0] = 1
                W = W + np.transpose(W)
                W.setdiag(0)
                return W

            elif kwargs['metric'] == 'cosine':
                # normalize the data first
                X_norm = np.power(np.sum(X*X, axis=1), 0.5)
                for i in range(n_samples):
                    X[i, :] = X[i, :]/max(1e-12, X_norm[i])
                # compute pairwise cosine distances
                D_cosine = np.dot(X, np.transpose(X))
                # sort the distance matrix D in descending order
                dump = np.sort(-D_cosine, axis=1)
                idx = np.argsort(-D_cosine, axis=1)
                idx_new = idx[:, 0:k+1]
                dump_new = -dump[:, 0:k+1]
                G = np.zeros((n_samples*(k+1), 3))
                G[:, 0] = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
                G[:, 1] = np.ravel(idx_new, order='F')
                G[:, 2] = np.ravel(dump_new, order='F')
                # build sparse affinity matrix W
                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                W[W > 0] = 1
                W = W + np.transpose(W)
                W.setdiag(0)
                return W

        elif kwargs['weight_mode'] == 'heat_kernel':
            t = kwargs['t']
            # compute pairwise euclidean distances
            D = pairwise_distances(X)
            # sort the distance matrix D in ascending order
            dump = np.sort(D, axis=1)
            idx = np.argsort(D, axis=1)
            idx_new = idx[:, 0:k+1]
            dump_new = dump[:, 0:k+1]
            # compute the pairwise heat kernel distances
            dump_heat_kernel = np.exp(-dump_new/(2*t*t))
            G = np.zeros((n_samples*(k+1), 3))
            G[:, 0] = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
            G[:, 1] = np.ravel(idx_new, order='F')
            G[:, 2] = np.ravel(dump_heat_kernel, order='F')
            # build sparse affinity matrix W
            W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            W = W + np.transpose(W)
            W.setdiag(0)
            return W

        elif kwargs['weight_mode'] == 'cosine':
            # normalize the data first
            X_norm = np.power(np.sum(X*X, axis=1), 0.5)
            for i in range(n_samples):
                    X[i, :] = X[i, :]/max(1e-12, X_norm[i])
            # compute pairwise cosine distances
            D_cosine = np.dot(X, np.transpose(X))
            # sort the distance matrix D in ascending order
            dump = np.sort(-D_cosine, axis=1)
            idx = np.argsort(-D_cosine, axis=1)
            idx_new = idx[:, 0:k+1]
            dump_new = -dump[:, 0:k+1]
            G = np.zeros((n_samples*(k+1), 3))
            G[:, 0] = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
            G[:, 1] = np.ravel(idx_new, order='F')
            G[:, 2] = np.ravel(dump_new, order='F')
            # build sparse affinity matrix W
            W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            W = W + np.transpose(W)
            W.setdiag(0)
            return W

    # Choose supervised neighborMode
    elif kwargs['neighbor_mode'] == 'supervised':
        k = kwargs['k']
        # Get y and the number of classes
        y = kwargs['y']
        label = np.unique(y)
        n_classes = np.unique(y).size
        # Construct the weight matrix W in a fisherScore way, W_ij = 1/n_l if yi = yj = l, otherwise W_ij = 0
        if kwargs['fisher_score'] is True:
            W = lil_matrix((n_samples, n_samples))
            for i in range(n_classes):
                class_idx = (y == label[i])
                class_idx_all = (class_idx[:, np.newaxis] & class_idx[np.newaxis, :])
                W[class_idx_all] = 1.0/np.sum(np.sum(class_idx))
            return W

        # Construct the weight matrix W in a relief way, NH(x) or NM(x,y) denotes a set of k nearest
        # points to x with the same class of x, or a different class (the class y), respectively. W_ij = 1 if i = j;
        # W_ij = -1/k if x_j \in NH(x_i); W_ij = 1/(c-1)k if xj \in NM(x_i, y)
        if kwargs['relief'] == True:
            # when xj in NH(xi)
            G = np.zeros((nSamples*(k+1), 3))
            idNow = 0
            for i in range(nLabel):
                classIdx = np.column_stack(np.where(y==label[i]))[:,0]
                D = pairwise_distances(data[classIdx,:])
                dump = np.sort(D, axis = 1)
                idx = np.argsort(D, axis = 1)
                idxNew = idx[:,0:k+1]
                nSmpClass = len(classIdx)*(k+1)
                G[idNow:nSmpClass+idNow, 0] = np.tile(classIdx,(k+1,1)).reshape(-1)
                G[idNow:nSmpClass+idNow, 1] = np.ravel(classIdx[idxNew[:]], order = 'F')
                G[idNow:nSmpClass+idNow, 2] = -1.0/k
                idNow = idNow + nSmpClass
            W1 = csc_matrix((G[:,2],(G[:,0],G[:,1])), shape = (nSamples, nSamples)).toarray()
            W1 = np.minimum(W1, np.transpose(W1))
            # when i = j
            for i in range(nSamples):
                W1[i,i] = 1
            # when xj in NM(xi,y)
            G = np.zeros((nSamples*(k), 3))
            idNow = 0
            for i in range(nLabel):
                classIdx1 = np.column_stack(np.where(y==label[i]))[:,0]
                classIdx2 = np.column_stack(np.where(y!=label[i]))[:,0]
                X1 = data[classIdx1,:]
                X2 = data[classIdx2,:]
                D = pairwise_distances(X1,X2)
                dump = np.sort(D, axis = 1)
                idx = np.argsort(D, axis = 1)
                idxNew = idx[:,0:k]
                nSmpClass = len(classIdx)*k
                G[idNow:nSmpClass+idNow, 0] = np.tile(classIdx1,(k,1)).reshape(-1)
                G[idNow:nSmpClass+idNow, 1] = np.ravel(classIdx2[idxNew[:]], order = 'F')
                G[idNow:nSmpClass+idNow, 2] = 1.0/((nLabel-1)*k)
                idNow = idNow + nSmpClass
            W2 = csc_matrix((G[:,2],(G[:,0],G[:,1])), shape = (nSamples, nSamples)).toarray()
            W2 = np.maximum(W2, np.transpose(W2))
            W = W1 + W2
            print 'test'
            return W

        if kwargs['weight_mode'] == 'binary':
            if kwargs['metric'] == 'euclidean':
                G = np.zeros((n_samples*(k+1), 3))
                id_now = 0
                for i in range(n_classes):
                    class_idx = np.column_stack(np.where(y==label[i]))[:, 0]
                    # compute pairwise euclidean distances for instances in class i
                    D = pairwise_distances(X[class_idx, :])
                    # sort the distance matrix D in ascending order for instances in class i
                    idx = np.argsort(D, axis=1)
                    idx_new = idx[:, 0:k+1]
                    n_smp_class = len(class_idx)*(k+1)
                    G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
                    G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                    G[id_now:n_smp_class+id_now, 2] = 1
                    id_now += n_smp_class
                # build sparse affinity matrix W
                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                W = W + np.transpose(W)
                W.setdiag(0)
                return W

            if kwargs['metric'] == 'cosine':
                # normalize the data first
                X_norm = np.power(np.sum(X*X, axis=1), 0.5)
                for i in range(n_samples):
                    X[i, :] = X[i, :]/max(1e-12, X_norm[i])
                G = np.zeros((n_samples*(k+1), 3))
                id_now = 0
                for i in range(n_classes):
                    class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                    # compute pairwise cosine distances for instances in class i
                    D_cosine = np.dot(X[class_idx, :], np.transpose(X[class_idx, :]))
                    # sort the distance matrix D in descending order for instances in class i
                    idx = np.argsort(-D_cosine, axis=1)
                    idx_new = idx[:, 0:k+1]
                    n_smp_class = len(class_idx)*(k+1)
                    G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
                    G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                    G[id_now:n_smp_class+id_now, 2] = 1
                    id_now += n_smp_class
                # build sparse affinity matrix W
                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                W = W + np.transpose(W)
                W.setdiag(0)
                return W

        elif kwargs['weight_mode'] == 'heat_kernel':
            G = np.zeros((n_samples*(k+1), 3))
            id_now = 0
            for i in range(n_classes):
                class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                # compute pairwise cosine distances for instances in class i
                D = pairwise_distances(X[class_idx, :])
                # sort the distance matrix D in ascending order for instances in class i
                dump = np.sort(D, axis=1)
                idx = np.argsort(D, axis=1)
                idx_new = idx[:, 0:k+1]
                dump_new = dump[:, 0:k+1]
                t = kwargs['t']
                # compute pairwise heat kernel distances for instances in class i
                dump_heat_kernel = np.exp(-dump_new/(2*t*t))
                n_smp_class = len(class_idx)*(k+1)
                G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
                G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                G[id_now:n_smp_class+id_now, 2] = np.ravel(dump_heat_kernel, order='F')
                id_now += n_smp_class
            # build sparse affinity matrix W
            W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            W = W + np.transpose(W)
            W.setdiag(0)
            return W

        elif kwargs['weight_mode'] == 'cosine':
            # normalize the data first
            X_norm = np.power(np.sum(X*X, axis=1), 0.5)
            for i in range(n_samples):
                X[i, :] = X[i, :]/max(1e-12, X_norm[i])
            G = np.zeros((n_samples*(k+1), 3))
            id_now = 0
            for i in range(n_classes):
                class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                # compute pairwise cosine distances for instances in class i
                D_cosine = np.dot(X[class_idx, :], np.transpose(X[class_idx, :]))
                # sort the distance matrix D in descending order for instances in class i
                dump = np.sort(-D_cosine, axis=1)
                idx = np.argsort(-D_cosine, axis=1)
                idx_new = idx[:, 0:k+1]
                dump_new = -dump[:, 0:k+1]
                n_smp_class = len(class_idx)*(k+1)
                G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
                G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                G[id_now:n_smp_class+id_now, 2] = np.ravel(dump_new, order='F')
                id_now = id_now + n_smp_class
            # build sparse affinity matrix W
            W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            W = W + np.transpose(W)
            W.setdiag(0)
            return W