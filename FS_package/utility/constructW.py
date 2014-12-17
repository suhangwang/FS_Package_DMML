import numpy as np
from scipy.sparse import *
from sklearn.metrics.pairwise import pairwise_distances

def constructW(data, **kwargs):
    """ Construct the weight matrix W
    If kwargs is none, then use the default parameter settings
    If kwargs is not none, construct the weight matrix according to parameters in kwargs
    Input
    ----------
    data: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a numpy array
    kwargs: {dictionary}
        Parameters to construct different weight matrix W:
            trueLabel: {numpy array}, shape (n_samples, 1)
                The parameter needed under the 'supervised' neighbor mode
            metric: {string}
                Choices for different distance measure
                'euclidean' - use euclidean distance
                 'cosine' - use cosine distance (default)
            neighborMode: {string}
                Indicates how to construct the graph
                'knn' - Put an edge between two nodes if and only if they are among the
                        k nearest neighbors of each other (default)
                'supervised' - Put an edge between two nodes if they belong to same class
                        and they are among the k nearest neighbors of each other
            weightMode: {string}
                Indicates how to assign weights for each edge in the graph
                'binary' - 0-1 weighting, every edge receives weight of 1 (default)
                'heatKernel' - If nodes i and j are connected, put weight W_ij = exp(-norm(x_i - x_j)/2t^2)
                                This weight mode can only be used under 'euclidean' metric and you are required
                                to provide the parameter t
                'cosine' - If nodes i and j are connected, put weight cosine(x_i,x_j).
                            Can only be used under 'cosine' metric
            k: {int}
                Choices for the number of neighbors (default k = 5)
            t: {float}
                Parameter for the 'heatKernel' weightMode
            fisherScore: {boolean}
                Indicates whether to build the weight matrix in a fisherScore way, in which W_ij = 1/n_l if yi = yj = l,
                otherwise W_ij = 0 (default fisherScore = false)
            relief: {boolean}
                Indicates whether to build the weight matrix in a relief way, NH(x) or NM(x,y) denotes a set of k nearest
                points to x with the same class of x, or a different class (the class y), respectively. W_ij = 1 if i = j;
                W_ij = -1/k if x_j \in NH(x_i); W_ij = 1/(c-1)k if xj \in NM(x_i, y) (default relief = false)
    Returns
    -------
    W: {array-like}, shape (n_samples, n_samples)
        Output weight matrix W, guaranteed to be a numpy array.
    """

    # Initialization part
    # Set default values for the parameter 'metric'
    if 'metric' not in kwargs.keys():
        kwargs['metric'] = 'cosine'

    # Set default values for the parameter 'neighborMode' and corresponding k
    if 'neighborMode' not in kwargs.keys():
        kwargs['neighborMode'] = 'knn'
    if kwargs['neighborMode'] == 'knn' and 'k' not in kwargs.keys():
        kwargs['k'] = 5
    if kwargs['neighborMode'] == 'supervised' and 'k' not in kwargs.keys():
        kwargs['k'] = 5
    if kwargs['neighborMode'] == 'supervised' and 'trueLabel' not in kwargs.keys():
        print ('Warning: trueLabel is required in the supervised neighborMode!!!')
        exit(0)

    # Set default values for the parameter 'weightMode' and corresponding t if in heatKernel weightMode
    if 'weightMode' not in kwargs.keys():
        kwargs['weightMode'] = 'binary'
    if kwargs['weightMode'] == 'heatKernel':
        if kwargs['metric'] != 'euclidean':
            kwargs['metric'] = 'euclidean'
        if 't' not in kwargs.keys():
            kwargs['t'] = 1
    elif kwargs['weightMode'] == 'cosine':
        if kwargs['metric'] != 'cosine':
            kwargs['metric'] = 'cosine'

    # Set default values for the parameter 'fisherScore' and 'relief' to be False
    if 'fisherScore' not in kwargs.keys():
        kwargs['fisherScore'] = False
    if 'relief' not in kwargs.keys():
        kwargs['relief'] = False

    # Get number of samples and parameter k
    nSamples = len(data)

    # Choose 'knn' neighborMode
    if kwargs['neighborMode'] == 'knn':
        k = kwargs['k']
        if kwargs['weightMode'] == 'binary':
            if kwargs['metric'] == 'euclidean':
                D = pairwise_distances(data)
                dump = np.sort(D, axis = 1)
                idx = np.argsort(D, axis = 1)
                idxNew = idx[:,0:k+1]
                dumpNew = dump[:,0:k+1]
                G = np.zeros((nSamples*(k+1), 3))
                G[:,0] = np.tile(np.arange(nSamples),(k+1,1)).reshape(-1)
                G[:,1] = np.ravel(idxNew, order = 'F')
                G[:,2] = np.ravel(dumpNew, order = 'F')
                W = csc_matrix((G[:,2],(G[:,0],G[:,1])), shape = (nSamples, nSamples))
                W[W>0] = 1
                W = W + np.transpose(W)
                W.setdiag(0)
                return W
            elif kwargs['metric'] == 'cosine':
                # 1. Normalize the data first
                dataNorm = np.power(np.sum(data*data, axis = 1), 0.5)
                for i in range(nSamples):
                    data[i,:] = data[i,:]/max(1e-12, dataNorm[i])
                # 2. Construct weight matrix W
                Dcosine = np.dot(data,np.transpose(data))
                dump = np.sort(-Dcosine, axis = 1)
                idx = np.argsort(-Dcosine, axis = 1)
                idxNew = idx[:,0:k+1]
                dumpNew = -dump[:,0:k+1]
                G = np.zeros((nSamples*(k+1), 3))
                G[:,0] = np.tile(np.arange(nSamples),(k+1,1)).reshape(-1)
                G[:,1] = np.ravel(idxNew, order = 'F')
                G[:,2] = np.ravel(dumpNew, order = 'F')
                W = csc_matrix((G[:,2],(G[:,0],G[:,1])), shape = (nSamples, nSamples))
                W[W>0] = 1
                W = W + np.transpose(W)
                W.setdiag(0)
                return W

        elif kwargs['weightMode'] == 'heatKernel':
            t = kwargs['t']
            D = pairwise_distances(data)
            dump = np.sort(D, axis = 1)
            idx = np.argsort(D, axis = 1)
            idxNew = idx[:,0:k+1]
            dumpNew = dump[:,0:k+1]
            dumpHeatKernel = np.exp(-dumpNew/(2*t*t))
            G = np.zeros((nSamples*(k+1), 3))
            G[:,0] = np.tile(np.arange(nSamples),(k+1,1)).reshape(-1)
            G[:,1] = np.ravel(idxNew, order = 'F')
            G[:,2] = np.ravel(dumpHeatKernel, order = 'F')
            W = csc_matrix((G[:,2],(G[:,0],G[:,1])), shape = (nSamples, nSamples))
            W = W + np.transpose(W)
            W.setdiag(0)
            return W

        elif kwargs['weightMode'] == 'cosine':
            # 1. Normalize the data first
            dataNorm = np.power(np.sum(data*data, axis = 1), 0.5)
            for i in range(nSamples):
                    data[i,:] = data[i,:]/max(1e-12, dataNorm[i])
            # 2. Construct weight matrix W
            Dcosine = np.dot(data,np.transpose(data))
            dump = np.sort(-Dcosine, axis = 1)
            idx = np.argsort(-Dcosine, axis = 1)
            idxNew = idx[:,0:k+1]
            dumpNew = -dump[:,0:k+1]
            G = np.zeros((nSamples*(k+1), 3))
            G[:,0] = np.tile(np.arange(nSamples),(k+1,1)).reshape(-1)
            G[:,1] = np.ravel(idxNew, order = 'F')
            G[:,2] = np.ravel(dumpNew, order = 'F')
            W = csc_matrix((G[:,2],(G[:,0],G[:,1])), shape = (nSamples, nSamples))
            W = W + np.transpose(W)
            W.setdiag(0)
            return W

    # Choose supervised neighborMode
    elif kwargs['neighborMode'] == 'supervised':
        k = kwargs['k']
        # Get the trueLabel and the number of classes
        y = kwargs['trueLabel']
        label = np.unique(y)
        nLabel = np.unique(y).size
        # Construct the weight matrix W in a fisherScore way, W_ij = 1/n_l if yi = yj = l, otherwise W_ij = 0
        if kwargs['fisherScore'] == True:
            print 'fisher'
            #W = np.zeros((nSamples, nSamples))
            W = lil_matrix((nSamples,nSamples))
            for i in range(nLabel):
                classIdx = (y==label[i])
                classIdxAll = (classIdx[:,np.newaxis] & classIdx[np.newaxis,:])
                W[classIdxAll] = 1.0/np.sum(np.sum(classIdx))
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

        if kwargs['weightMode'] == 'binary':
            if kwargs['metric'] == 'euclidean':
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
                    G[idNow:nSmpClass+idNow, 2] = 1
                    idNow = idNow + nSmpClass
                W = csc_matrix((G[:,2],(G[:,0],G[:,1])), shape = (nSamples, nSamples))
                W = W + np.transpose(W)
                W.setdiag(0)
                return W
            if kwargs['metric'] == 'cosine':
                # 1. Normalize the data first
                dataNorm = np.power(np.sum(data*data, axis = 1), 0.5)
                for i in range(nSamples):
                    data[i,:] = data[i,:]/max(1e-12, dataNorm[i])
                # 2. Construct weight matrix W
                G = np.zeros((nSamples*(k+1), 3))
                idNow = 0
                for i in range(nLabel):
                    classIdx = np.column_stack(np.where(y==label[i]))[:,0]
                    Dcosine = np.dot(data[classIdx,:],np.transpose(data[classIdx,:]))
                    dump = np.sort(-Dcosine, axis = 1)
                    idx = np.argsort(-Dcosine, axis = 1)
                    idxNew = idx[:,0:k+1]
                    dumpNew = -dump[:,0:k+1]
                    nSmpClass = len(classIdx)*(k+1)
                    G[idNow:nSmpClass+idNow, 0] = np.tile(classIdx,(k+1,1)).reshape(-1)
                    G[idNow:nSmpClass+idNow, 1] = np.ravel(classIdx[idxNew[:]], order = 'F')
                    G[idNow:nSmpClass+idNow, 2] = 1
                    idNow = idNow + nSmpClass
                W = csc_matrix((G[:,2],(G[:,0],G[:,1])), shape = (nSamples, nSamples))
                W = W + np.transpose(W)
                W.setdiag(0)
                return W

        elif kwargs['weightMode'] == 'heatKernel':
            G = np.zeros((nSamples*(k+1), 3))
            idNow = 0
            for i in range(nLabel):
                classIdx = np.column_stack(np.where(y==label[i]))[:,0]
                D = pairwise_distances(data[classIdx,:])
                dump = np.sort(D, axis = 1)
                idx = np.argsort(D, axis = 1)
                idxNew = idx[:,0:k+1]
                dumpNew = dump[:,0:k+1]
                t = kwargs['t']
                dumpHeatKernel = np.exp(-dumpNew/(2*t*t))
                nSmpClass = len(classIdx)*(k+1)
                G[idNow:nSmpClass+idNow, 0] = np.tile(classIdx,(k+1,1)).reshape(-1)
                G[idNow:nSmpClass+idNow, 1] = np.ravel(classIdx[idxNew[:]], order = 'F')
                G[idNow:nSmpClass+idNow, 2] = np.ravel(dumpHeatKernel, order = 'F')
                idNow = idNow + nSmpClass
            W = csc_matrix((G[:,2],(G[:,0],G[:,1])), shape = (nSamples, nSamples))
            W = W + np.transpose(W)
            W.setdiag(0)
            return W

        elif kwargs['weightMode'] == 'cosine':
            # 1. Normalize the data first
            dataNorm = np.power(np.sum(data*data, axis = 1), 0.5)
            for i in range(nSamples):
                data[i,:] = data[i,:]/max(1e-12, dataNorm[i])
			# 2. Construct weight matrix W
            G = np.zeros((nSamples*(k+1), 3))
            idNow = 0
            for i in range(nLabel):
                classIdx = np.column_stack(np.where(y==label[i]))[:,0]
                Dcosine = np.dot(data[classIdx,:],np.transpose(data[classIdx,:]))
                dump = np.sort(-Dcosine, axis = 1)
                idx = np.argsort(-Dcosine, axis = 1)
                idxNew = idx[:,0:k+1]
                dumpNew = -dump[:,0:k+1]
                nSmpClass = len(classIdx)*(k+1)
                G[idNow:nSmpClass+idNow, 0] = np.tile(classIdx,(k+1,1)).reshape(-1)
                G[idNow:nSmpClass+idNow, 1] = np.ravel(classIdx[idxNew[:]], order = 'F')
                G[idNow:nSmpClass+idNow, 2] = np.ravel(dumpNew, order = 'F')
                idNow = idNow + nSmpClass
            W = csc_matrix((G[:,2],(G[:,0],G[:,1])), shape = (nSamples, nSamples))
            W = W + np.transpose(W)
            W.setdiag(0)
            return W