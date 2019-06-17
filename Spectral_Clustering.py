import numpy as np
from sklearn.cluster import SpectralClustering

ADJ_MAT_DESIGN = 'A'

def compute_spectral_clustering(adj_mat, num_of_clusters):
    N = np.shape(adj_mat)[0]
    assert np.shape(adj_mat) == (N, N)
    sc = SpectralClustering(n_clusters=num_of_clusters, affinity='precomputed', n_init=100, assign_labels='kmeans')
    assignments = sc.fit_predict(adj_mat)
    assert np.shape(assignments) == (N,)
    return assignments

# For now, just construct adjacency matrix layout by layout
def compute_adj_mat_from_CSI(CSI_mat):
    number_of_links = np.shape(CSI_mat)[0]
    assert np.shape(CSI_mat) == (number_of_links, number_of_links)
    if(ADJ_MAT_DESIGN=='A'):
        adj_mat = np.maximum(CSI_mat, np.transpose(CSI_mat))
        adj_mat = adj_mat * ((np.eye(number_of_links)<1).astype(float))
    elif(ADJ_MAT_DESIGN=='B'):
        
    assert np.shape(adj_mat) == (number_of_links, number_of_links)
    return adj_mat