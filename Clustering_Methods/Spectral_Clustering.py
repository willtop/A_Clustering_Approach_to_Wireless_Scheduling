import numpy as np
from sklearn.cluster import KMeans
import sys
import visualize_clusters
sys.path.append("../Utilities_Research/")
import utils

VISUALIZE = False
# For now, just construct adjacency matrix layout by layout
def construct_adj_mat(channel_losses_mat):
    number_of_links = np.shape(channel_losses_mat)[0]
    assert np.shape(channel_losses_mat) == (number_of_links, number_of_links)
    adj_mat = np.maximum(channel_losses_mat, np.transpose(channel_losses_mat)) # no need to clear-off the diagonal
    assert np.shape(adj_mat) == (number_of_links, number_of_links)
    assert np.all(np.transpose(adj_mat) == adj_mat) # ensure the symmetry
    return adj_mat

# For now, process one layout at a time
def clustering(layout, channel_losses_mat, n_links_on):
    N = np.shape(channel_losses_mat)[0]
    assert np.shape(channel_losses_mat) == (N, N)
    assert np.shape(layout) == (N, 4)
    adj_mat = construct_adj_mat(channel_losses_mat)
    assert np.shape(adj_mat) == (N, N)
    laplace_mat = np.diag(np.sum(adj_mat, axis=1)) - adj_mat
    eig_vals, eig_vecs = np.linalg.eig(laplace_mat)
    smallest_eigval_indices = np.argsort(eig_vals)[1:n_links_on] # no need to take the zero eigen value
    eig_vecs_selected = eig_vecs[:, smallest_eigval_indices]
    km = KMeans(n_clusters=n_links_on, n_init=10).fit(eig_vecs_selected)
    cluster_assignments = km.labels_
    assert np.shape(cluster_assignments) == (N, )
    if (VISUALIZE):
        visualize_clusters.visualize_layout_clusters(layout, cluster_assignments, "Spectral Clustering")
    return cluster_assignments

def scheduling(channel_losses_mat, cluster_assignments):
    N = np.shape(channel_losses_mat)[0]
    assert np.shape(channel_losses_mat) == (N, N)
    assert np.shape(cluster_assignments) == (N, )
    allocations = np.zeros(N)
    # Select one strongest link from each cluster to schedule
    n_links_on = np.max(cluster_assignments)+1
    for i in range(n_links_on):
        links_in_the_cluster = np.where(cluster_assignments == i)[0]
        strongest_link_in_the_cluster = links_in_the_cluster[np.argmax(np.diag(channel_losses_mat)[links_in_the_cluster])]
        assert allocations[strongest_link_in_the_cluster] == 0, "having duplicate entry appearence across clusters"
        allocations[strongest_link_in_the_cluster] = 1
    return allocations