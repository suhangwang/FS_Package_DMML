import scipy.io
import sklearn.cluster
import numpy as np
from utility.constructW import constructW
from utility.unsupervised_evaluation import evaluation

def FisherScore(data, **kwargs):
    """
    This function implement the FisherScore function
    1. Construct the weight matrix W if it is not specified
    2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
    3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
    4. FisherScore for the r-th feature is Lr = (fr_hat'*L*fr_hat)/*(fr_hat'*D*fr_hat)

    Input
    ----------
    data : {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a numpy array
    kwargs : {dictionary}
        W : {numpy array}, shape (n_samples, n_samples)
        Input weight matrix

    Reference:
        He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS. 2005.
    """
    N,d = data.shape
    X = data
    if 'W' not in kwargs.keys():
        W = constructW(data)

    W = kwargs['W']
    D = np.sum(W, axis=1)
    L = W
    tmp = np.dot(np.transpose(D), data)
    D = np.diag(D)
    Xt = np.transpose(X)
    t1 = np.transpose(np.dot(Xt,D))
    t2 = np.transpose(np.dot(Xt,L))
    DPrime = np.sum(np.multiply(t1,X),0) - np.multiply(tmp,tmp)/np.sum(np.diag(D))
    LPrime = np.sum(np.multiply(t2,X),0) - np.multiply(tmp,tmp)/np.sum(np.diag(D))
    DPrime[DPrime < 1e-12] = 10000
    score = np.multiply(LPrime,1/DPrime)
    one = np.ones((N,d))
    scoreReciprocal = np.true_divide(one,score)
    
    score = np.subtract(scoreReciprocal,one)
    score = np.transpose(score)
    return score

def featureRanking(score):
    ind = np.argsort(score,0)
    return ind[::-1]

def main():
    # load matlab data
    mat = scipy.io.loadmat('data/LUNG.mat')
    Lable = mat['L']    # label
    Lable = Lable[:,0]
    X = mat['M']    # data
    N,d = X.shape

    W = constructW(X)

    # feature weight learning / feature selection
    score = FisherScore(X, W = W)
    IND = featureRanking(score)

    # evalaution
    numFea = 100
    selectedFeatures = X[:,IND[0:numFea]]

    ARI, NMI, ACC, predictLabel = evaluation(selectedFeatures = selectedFeatures, C=5, Y=Lable)
    print ARI
    print NMI
    print ACC
    print predictLabel.astype(int)

if __name__=='__main__':
    main()