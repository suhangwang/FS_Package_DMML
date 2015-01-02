import sys
import numpy as np
import sklearn.utils.linear_assignment_ as la
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans


def best_map(l1, l2):
    """
    Permute labels of L2 to match L1 as much as possible
    """
    if len(l1) != len(l2):
        print >>sys.stderr, "L1.shape must == L2.shape"

    label1 = np.unique(l1)
    n_class1 = len(label1)

    label2 = np.unique(l2)
    n_class2 = len(label2)

    n_class = max(n_class1, n_class2)
    G = np.zeros((n_class, n_class))

    for i in range(0, n_class1):
        for j in range(0, n_class2):
            ss = l1 == label1[i]
            tt = l2 == label2[j]
            G[i, j] = np.count_nonzero(ss & tt)

    A = la.linear_assignment(-G)

    new_l2 = np.zeros(l2.shape)
    for i in range(0, n_class2):
        new_l2[l2 == label2[A[i][1]]] = label1[A[i][0]]
    return new_l2.astype(int)


def evaluation(selected_features, n_clusters, y):
    """
    Calculate ACC and NMI of the selected features
    Input
    ----------
        selectedFeatures: {numpy array}, shape (n_samples, n_selectedFeatures}
            data of the selectedFeatures
        C: {int}
            number of clusters
        Y: {numpy array}, shape (n_samples, 1)
            actual labels
    Output
    ----------
        Adjusted Rand Index: {float}
        Normalized Mutual Information: {float}
        Accuracy: {float}
    """
    k_means = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                     tol=0.0001, precompute_distances=True, verbose=0,
                     random_state=None, copy_x=True, n_jobs=1)

    k_means.fit(selected_features)
    y_predict = k_means.labels_

    # calculate ARI
    ari = adjusted_rand_score(y, y_predict)

    # calculate NMI
    nmi = normalized_mutual_info_score(y, y_predict)

    # calculate ACC
    y_permuted_predict = best_map(y, y_predict)
    acc = accuracy_score(y, y_permuted_predict)

    return ari, nmi, acc