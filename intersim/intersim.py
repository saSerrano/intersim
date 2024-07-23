import os
import numpy as np
from copy import copy, deepcopy
from sklearn.cluster import KMeans

from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy

def rl(task_name):
    '''
    Perform Q-learning

    Check the following link for an example:
    https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/grid_world_td.py
    
    '''
    np.random.seed()

def cluster(m,n_clusters,mode):
    '''
    Cluster the elements in each row (or column) and return the assigned labels.
    '''
    assert len(m.shape) == 2 and n_clusters >= 2 and mode in ['rows','columns']
    
    a = None
    if mode == 'rows':
        a = np.copy(m)
    else:
        a = np.transpose(np.copy(m))

    labels = np.zeros(a.shape,dtype=np.int64)
    for i in range(a.shape[0]):
        # Cluster row elements
        x = np.copy(a[i,:]).reshape((a.shape[1],1))
        km = KMeans(n_clusters=n_clusters, random_state=0).fit(x)

        # Sort clusters by their centroid
        l, c = km.labels_, km.cluster_centers_.flatten()
        cluster_ids = [(j,c[j]) for j in range(c.shape[0])].sort(key=lambda y : y[1])
        id_map = {}
        for j in range(len(cluster_ids)):
            id_map[cluster_ids[0]] = j
        for j in range(a.shape[1]):
            labels[i,j] = id_map[l[j]]
    
    if mode == 'rows':
        return labels
    else:
        return np.transpose(labels)

def label_histograms(m,n_labels,mode):
    '''
    Builds a matrix in which each row represents the histogram
    of labels assigned in the rows or columns of input matrix m.

    INPUT

    m (np-array): input matrix.
    n_labels (int): number of different labels present in 'm'.
    mode (str): string indicating wheather labels should counted
    over rows or columns.

    OUTPUT

    f (np-array): matrix containing the label histograms (one in
    each row).
    '''
    assert len(m.shape) == 2 and n_labels >= 2 and mode in ['rows','columns']
    a = None
    if mode == 'rows':
        a = np.copy(m)
    else:
        a = np.transpose(np.copy(m))

    f = np.zeros((a.shape[0],n_labels),dtype=np.float64)
    for i in range(a.shape[0]):
        for j in range(n_labels):
            f[i,j] = float(a[i,:].tolist().count(j)) / float(a.shape[1])
    
    return f

def intersection_mats(q1,q2,n_clusters=2):
    '''
    DESCRIPTION

    This function computes the action and state intersection
    matrices for a pair of Q-tables.

    INPUT

    q1 (np-array): q-table 1.
    q2 (np-array): q-table 2.
    n_clusters (int): number of clusters.

    OUTPUT

    inter_a (np-array): actions intersection matrix.
    inter_s (np-array): states intersection matrix.
    '''
    # Compute the frequency matrices 
    fa1 = label_histograms(cluster(q1,n_clusters,'rows'),n_clusters,'columns')
    fs1 = label_histograms(cluster(q1,n_clusters,'columns'),n_clusters,'rows')
    fa2 = label_histograms(cluster(q2,n_clusters,'rows'),n_clusters,'columns')
    fs2 = label_histograms(cluster(q2,n_clusters,'columns'),n_clusters,'rows')
    
    # Compute intersection matrices
    inter_a, inter_s = np.zeros((fa1.shape[0],fa2.shape[0]),dtype=np.float64), np.zeros((fs1.shape[0],fs2.shape[0]),dtype=np.float64)
    for i in range(inter_a.shape[0]):
        for j in range(inter_a.shape[1]):
            # Intersection of histograms
            tmp = np.concatenate((np.copy(fa1[i,:]).reshape(1,n_clusters),np.copy(fa2[i,:]).reshape(1,n_clusters)),axis=0)
            inter_a[i,j] = np.sum(np.min(tmp,axis=0))
    for i in range(inter_s.shape[0]):
        for j in range(inter_s.shape[1]):
            # Intersection of histograms
            tmp = np.concatenate((np.copy(fs1[i,:]).reshape(1,n_clusters),np.copy(fs2[i,:]).reshape(1,n_clusters)),axis=0)
            inter_s[i,j] = np.sum(np.min(tmp,axis=0))

    return inter_a, inter_s

def intertask_similarity(inter_a,inter_s):
    '''
    DESCRIPTION

    Computes the mean/frobenius-based similarity scores
    of two Q-tables, given their action and state inter-
    section matrices.

    INPUT

    inter_a (np-array): action intersection matrix.
    inter_s (np-array): state intersection matrix.

    OUTPUT

    mean_a (np-array): mean-based action similarity score.
    mean_s (np-array): mean-based state similarity score.
    frob_a (np-array): frobenius-based action similarity score.
    frob_s (np-array): frobenius-based state similarity score.
    '''
    mean_a = float(np.mean(inter_a))
    mean_s = float(np.mean(inter_s))
    frob_a = np.linalg.norm(inter_a,'fro')
    frob_s = np.linalg.norm(inter_s,'fro')
    return mean_a, mean_s, frob_a, frob_s

def transfer(src_q,inter_a,inter_s):
    '''
    DESCRIPTION

    Computes the Q-table for the target task, trans-
    ferred from the source task Q-table.

    INPUT

    src_q   (np-array): source task Q-table.
    inter_a (np-array): action intersection matrix.
    inter_s (np-array): state intersection matrix.

    OUTPUT

    tgt_q   (np-array): transferred target task Q-table.
    '''
    # Select the most similar source state-action pair to each
    # target state-action pair
    tgt_q = np.zeros((inter_s.shape[1],inter_a.shape[1]),dtype=np.float64)
    for i in range(tgt_q.shape[0]):
        ms_state = np.argmax(inter_s[:,i])
        for j in range(tgt_q.shape[1]):
            ms_action = np.argmax(inter_a[:,j])
            tgt_q[i,j] = copy(src_q[ms_state,ms_action])
    
    return tgt_q

def main():
    # Learn Q tables
    q1 = rl('task 1')
    q2 = rl('task 2')

    # Compute instersection matrices
    n_clusters = 2
    inter_a, inter_s = intersection_mats(q1,q2,n_clusters)

    # Measure similarity scores
    ma,ms,fa,fs = intertask_similarity(inter_a,inter_s)

    # Transfer Q-values
    tl_q2 = transfer(q1,inter_a,inter_s)

if __name__ == '__main__':
    main()