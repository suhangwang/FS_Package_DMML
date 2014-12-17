import scipy.io
import numpy as np
from utility.constructW import constructW
from utility.unsupervised_evaluation import evaluation

def LapScore(X, **kwargs):
    """
    This function implement the LapScore function
    1. Construct the weight matrix W if it is not specified
    2. For the r-th feature, we define fr = data(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
    3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
    4. Laplacian score for the r-th feature is Lr = (fr_hat'*L*fr_hat)/*(fr_hat'*D*fr_hat)

    Input
    ----------
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a numpy array
    kwargs: {dictionary}
        W: {numpy array}, shape (n_samples, n_samples)
        Input weight matrix

    Reference:
        He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS. 2005.
    """
    N,d = X.shape
    if 'W' not in kwargs.keys():
        W = constructW(X)

    W = kwargs['W']
    D = np.sum(W, axis=1)
    L = W
    tmp = np.dot(np.transpose(D), X)
    D = np.diag(D)
    Xt = np.transpose(X)
    t1 = np.transpose(np.dot(Xt,D))
    t2 = np.transpose(np.dot(Xt,L))
    DPrime = np.sum(np.multiply(t1,X),0) - np.multiply(tmp,tmp)/np.sum(np.diag(D))
    LPrime = np.sum(np.multiply(t2,X),0) - np.multiply(tmp,tmp)/np.sum(np.diag(D))
    DPrime[DPrime < 1e-12] = 10000
    score = np.multiply(LPrime,1/DPrime)
    score = np.transpose(score)
    return score

def featureRanking(score):
    ind = np.argsort(score,0)
    return ind[::-1]

def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/USPS.mat')
    label = mat['gnd']    # label
    label = label[:,0]
    X = mat['fea']    # data
    N,d = X.shape
    X = X.astype(float)

    # normalize feature first
    # dataNorm = np.power(np.sum(X*X, axis = 1), 0.5)
    # for i in range(N):
    #     X[i,:] = X[i,:]/max(1e-12, dataNorm[i])

    # construct the weight matrix
    kwargs = {"metric": "euclidean","neighborMode": "knn","weightMode": "heatKernel","k": 5, 't': 1}
    W = constructW(X,**kwargs)

    # feature weight learning / feature selection
    score = LapScore(X, W = W)
    idx = featureRanking(score)

    # evaluation
    numFea = 100
    selectedFeatures = X[:,idx[0:numFea]]
    ARI, NMI, ACC, predictLabel = evaluation(selectedFeatures = selectedFeatures, C=10, Y=label)
    print ARI
    print NMI
    print ACC
    #print predictLabel.astype(int)

if __name__=='__main__':
    main()