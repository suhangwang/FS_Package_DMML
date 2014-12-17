import scipy.io
import numpy as np
from utility.constructW import constructW
from utility.supervised_evaluation import evaluation_leaveOneLabel
from utility.supervised_evaluation import evaluation_split

def FisherScore(X, W):
    """
    This function implement the FisherScore function
    1. Construct the weight matrix W in fisherScore way
    2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
    3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
    4. FisherScore for the r-th feature is Lr = (fr_hat'*L*fr_hat)/*(fr_hat'*D*fr_hat)

    Input
    ----------
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a numpy array
    kwargs : {dictionary}
        W: {numpy array}, shape (n_samples, n_samples)
        Input weight matrix
    """
    N,d = X.shape
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
    mat = scipy.io.loadmat('../data/COIL20.mat')
    label = mat['gnd']    # label
    label = label[:,0]
    X = mat['fea']    # data
    N,d = X.shape
    X = X.astype(float)
    # Construct weight matrix W in a fisherScore way
    kwargs = {"neighborMode": "supervised","fisherScore": True, 'trueLabel': label}
    W = constructW(X,**kwargs)

    # feature weight learning / feature selection
    score = FisherScore(X, W)
    idx = featureRanking(score)

    # evalaution
    numFea = 100
    selectedFeatures = X[:,idx[0:numFea]]
    ACC = evaluation_split(selectedFeatures = selectedFeatures, Y=label)
    print ACC

if __name__=='__main__':
    main()