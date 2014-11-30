import sys
import numpy as np
import sklearn.utils.linear_assignment_ as la
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans

def bestMap(L1, L2):
    """
    Permute labels of L2 to match L1 as good as possible
    """
    if len(L1) != len(L2):
        print >>sys.stderr, "L1.shape must == L2.shape"

    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)

    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass,nClass))

    for i in range(0,nClass1):
        for j in range(0,nClass2):
            ss = L1 == Label1[i]
            tt = L2 == Label2[j]
            # print (ss & tt)
            #G[i,j] = len(str.find(ss & tt))
            G[i,j] = np.count_nonzero(ss & tt)

    A = la.linear_assignment(-G)

    newL2 = np.zeros(L2.shape)
    for i in range(0,nClass2):
        newL2[L2 == Label2[A[i][1]]] = Label1[A[i][0]]
    return newL2.astype(int)

def evaluation(selectedFeatures, C, Y):
    """
    Calculate ACC and NMI of the selected features
    Input:
        C: number of clusters
        Y: actual label
    """
    kmeans = KMeans(n_clusters=C, init='k-means++', n_init=10, max_iter=300,
                            tol=0.0001, precompute_distances=True, verbose=0,
                            random_state=None, copy_x=True, n_jobs=1)
    kmeans.fit(selectedFeatures)
    predLabel = kmeans.labels_

    # calculate NMI and ARI
    NMI = normalized_mutual_info_score(Y, predLabel)
    ARI = adjusted_rand_score(Y,predLabel)

    # calculate ACC
    permutedPredLabel = bestMap(Y, predLabel)
    ACC = accuracy_score(Y, permutedPredLabel)

    return ARI, NMI, ACC, permutedPredLabel