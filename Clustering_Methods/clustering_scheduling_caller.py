# For any reproduce, further research or development, please kindly cite our SPAWC conference paper: 
# @Conference{cluster_schedule, 
#             author = "W. Cui and W. Yu", 
#             title = "A Clustering Approach to Wireless Scheduling", 
#             booktitle = "IEEE Workshop Signal Process. Advances Wireless Commun. (SPAWC)", 
#             year = 2020, 
#             month = may }
#
# This is a script that serves as a caller invoking different clustering scripts based on a string parameter.

import numpy as np
import Spectral_Clustering
import Hierarchical_Clustering
import K_Means
import Hierarchical_Clustering_EqualSize
import K_Means_EqualSize

# For spectral clustering & Hierarchical clustering and its variant
def adjacency_mat_based_scheduling(adj_mat, cluster_assignments):
    N = np.shape(adj_mat)[0]
    assert np.shape(adj_mat) == (N, N)
    assert np.shape(cluster_assignments) == (N,)
    allocations = np.zeros(N)
    # Select one strongest link from each cluster to schedule
    n_links_on = np.max(cluster_assignments) + 1
    for i in range(n_links_on):
        links_in_the_cluster = np.where(cluster_assignments == i)[0]
        if (np.size(links_in_the_cluster) == 1):
            # Just schedule that only link in the cluster (the following code dimension won't be enough)
            allocations[links_in_the_cluster[0]] = 1
        else:
            sub_adj_mat = adj_mat[np.ix_(links_in_the_cluster, links_in_the_cluster)]
            # Besides consider how it's interferencing with links within its own cluster, also count itself channel strength in
            link_reduced_index_to_schedule = np.argmax(np.sum(sub_adj_mat, axis=1))
            link_index_to_schedule = links_in_the_cluster[link_reduced_index_to_schedule]
            assert allocations[link_index_to_schedule] == 0, "having duplicate entry appearence across clusters"
            allocations[link_index_to_schedule] = 1
    return allocations

# For K-means clustering & its variant
def GLI_based_scheduling(layout, centroids, cluster_assignments):
    N = np.shape(layout)[0]
    assert np.shape(layout) == (N, 4)
    layout_midpoints = np.stack(((layout[:, 0] + layout[:, 2]) / 2, (layout[:, 1] + layout[:, 3]) / 2), axis=1)
    assert np.shape(layout_midpoints) == (N, 2)
    assert np.shape(cluster_assignments) == (N,)
    n_links_on = np.max(cluster_assignments) + 1
    assert np.shape(centroids) == (n_links_on, 2)
    allocations = np.zeros(N)
    # Select the link closest to each centroid to schedule (just O(N) computation)
    for i in range(n_links_on):
        links_in_the_cluster = np.where(cluster_assignments == i)[0]
        if(np.size(links_in_the_cluster) == 1):
            # Just schedule that only link in the cluster (the following code dimension won't be enough)
            allocations[links_in_the_cluster[0]] = 1
        else:
            # See which link is the closest to the centroid of this cluster
            link_reduced_index_to_schedule = np.argmin(np.linalg.norm(layout_midpoints[links_in_the_cluster]-centroids[i],axis=1))
            link_index_to_schedule = links_in_the_cluster[link_reduced_index_to_schedule]
            assert allocations[link_index_to_schedule] == 0, "having duplicate entry appearence across clusters"
            allocations[link_index_to_schedule] = 1
    return allocations

def clustering_and_scheduling(layout, channel_losses_mat, n_links_on, clustering_method):
    N = np.shape(layout)[0]
    assert np.shape(layout) == (N, 4)
    assert np.shape(channel_losses_mat) == (N, N)
    assert 1 <= n_links_on <= N
    if (clustering_method == "Spectral Clustering"):
        clusters_one_layout = Spectral_Clustering.clustering(layout, channel_losses_mat, n_links_on)
        adj_mat = Spectral_Clustering.construct_adj_mat(channel_losses_mat)
        allocs_one_layout = adjacency_mat_based_scheduling(adj_mat, clusters_one_layout)
    elif (clustering_method == "Hierarchical Clustering"):
        clusters_one_layout = Hierarchical_Clustering.clustering(layout, channel_losses_mat, n_links_on)
        adj_mat = Spectral_Clustering.construct_adj_mat(channel_losses_mat)
        allocs_one_layout = adjacency_mat_based_scheduling(adj_mat, clusters_one_layout)
    elif (clustering_method == "K-Means"):
        clusters_one_layout, centroids_one_layout = K_Means.clustering(layout, n_links_on)
        allocs_one_layout = GLI_based_scheduling(layout, centroids_one_layout, clusters_one_layout)
    elif (clustering_method == "Hierarchical Clustering EqualSize"):
        clusters_one_layout = Hierarchical_Clustering_EqualSize.clustering(layout, channel_losses_mat, n_links_on)
        adj_mat = Spectral_Clustering.construct_adj_mat(channel_losses_mat)
        allocs_one_layout = adjacency_mat_based_scheduling(adj_mat, clusters_one_layout)
    elif (clustering_method == "K-Means EqualSize"):
        clusters_one_layout, centroids_one_layout = K_Means_EqualSize.clustering(layout, n_links_on)
        allocs_one_layout = GLI_based_scheduling(layout, centroids_one_layout, clusters_one_layout)
    else:
        print("Invalid clustering method name: {}!".format(clustering_method))
        exit(1)
    return clusters_one_layout, allocs_one_layout
