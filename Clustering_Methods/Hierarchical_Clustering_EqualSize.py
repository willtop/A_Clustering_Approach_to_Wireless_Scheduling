# Hierarchical Clustering with equal size variation for clustering interferencing links
# This is inspired by: http://jmonlong.github.io/Hippocamplus/2018/06/09/cluster-same-size/

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../Utilities/")
import utils

VISUALIZE = False

# For now, just construct adjacency matrix layout by layout
def construct_adj_mat(channel_losses_mat):
    number_of_links = np.shape(channel_losses_mat)[0]
    # Empty diagonal entries since pair itself can't be regarded as very close in clustering
    crosslink_channel_losses_mat = utils.get_crosslink_channel_losses(channel_losses_mat)
    assert np.shape(crosslink_channel_losses_mat) == (number_of_links, number_of_links)
    adj_mat = np.maximum(crosslink_channel_losses_mat, np.transpose(crosslink_channel_losses_mat)) # no need to clear-off the diagonal
    assert np.shape(adj_mat) == (number_of_links, number_of_links)
    assert np.all(np.transpose(adj_mat) == adj_mat) # ensure the symmetry
    return adj_mat

def compute_1st_cluster_size_target(n_datapoints, n_clusters):
    assert n_clusters > 0
    # Compute desired cluster sizes
    smaller_cluster_size = int(np.floor(n_datapoints / n_clusters))
    if smaller_cluster_size * n_clusters < n_datapoints:
        smaller_cluster_size += 1
    return smaller_cluster_size

# Use the "average" linkage; merge B into A
def update_clusters(adj_mat, index_A, index_B, current_clusters):
    cluster_sizes = [len(cluster) for cluster in current_clusters]
    assert cluster_sizes[index_A]!=0 and cluster_sizes[index_B]!=0
    # Compute the weighted average for adjacencies connecting cluster A
    adj_mat[index_A, :] = (adj_mat[index_A, :]*cluster_sizes[index_A] + adj_mat[index_B, :]*cluster_sizes[index_B])/(cluster_sizes[index_A]+cluster_sizes[index_B])
    adj_mat[:, index_A] = (adj_mat[:, index_A]*cluster_sizes[index_A] + adj_mat[:, index_B]*cluster_sizes[index_B])/(cluster_sizes[index_A]+cluster_sizes[index_B])
    # Clear the cluster indexed at B since it's merged
    adj_mat[index_B, :] = 0
    adj_mat[:, index_B] = 0
    # Clear the new non-zero diagonal entry at cluster indexed at A
    adj_mat[index_A, index_A] = 0
    assert np.all(np.diag(adj_mat)==0)
    current_clusters[index_A] += current_clusters[index_B]
    current_clusters[index_B] = []
    return current_clusters, adj_mat

    
# Process one layout at a time
def clustering(layout, channel_losses_mat, n_links_on):
    N = np.shape(layout)[0]
    assert np.shape(channel_losses_mat) == (N, N)
    assert np.shape(layout) == (N,4)
    adj_mat = construct_adj_mat(channel_losses_mat)
    assert np.shape(adj_mat) == (N, N)
    cluster_assignments = -np.ones(N, dtype=np.int8)
    cluster_size_target = compute_1st_cluster_size_target(N, n_links_on)
    cluster_id = 0
    n_points_left = N
    current_clusters = [[i] for i in range(N)]
    while n_points_left > 0:
        while max([len(cluster) for cluster in current_clusters])<cluster_size_target:
            # This is where the cluster forming takes place
            entity_A_index, entity_B_index = np.unravel_index(np.argmax(adj_mat), (N, N), order='C')
            # modify the adjacency matrix in place, since the distance would be changed eitherway
            # This modification would still hold even after one cluster fullfilled
            current_clusters, adj_mat = update_clusters(adj_mat, entity_A_index, entity_B_index, current_clusters)
        cluster_formed = np.argmax(np.array([len(cluster) for cluster in current_clusters]))
        cluster_assignments[current_clusters[cluster_formed]] = cluster_id
        n_points_left -= len(current_clusters[cluster_formed])
        cluster_id += 1
        # Clear the cluster just fulfilled
        current_clusters[cluster_formed] = []
        adj_mat[cluster_formed, :] = 0
        adj_mat[:, cluster_formed] = 0
        # Check if all clusters are exhausted
        if cluster_id == n_links_on:
            assert n_points_left == 0
            break
        # There is a chance points are exhausted while not having enough clusters as desired
        # Just go with having fewer clusters, which would better explain the model
        if n_points_left == 0:
            break
        # Update the cluster size targets since the cluster just formed might be larger than desired
        cluster_size_target = compute_1st_cluster_size_target(n_points_left, n_links_on-cluster_id)
    assert np.shape(cluster_assignments)==(N,)
    if (VISUALIZE):
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Hierarchical Clustering {} Equal-Size Clusters Visualization".format(n_links_on))
        utils.plot_stations_on_layout(ax, layout)
        utils.plot_clusters_on_layout(ax, layout, cluster_assignments)
        plt.tight_layout()
        plt.show()
    return cluster_assignments
