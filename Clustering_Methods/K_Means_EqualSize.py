# K-Means with equal size variation for clustering interferencing links
# This is inspired by: https://elki-project.github.io/tutorial/same-size_k_means, but with simplification
# To ensure the viability of using K-Means, gonna simply use the mid-point of each transceiver pair as the representative point of the link

import numpy as np
from sklearn.cluster import KMeans
import visualize_clusters

VISUALIZE = False
# If assignment entry is -1, then it's not assigned yet
def update_centroid(centroids, new_point, new_cluster_assigned, cluster_sizes):
    centroid_old_loc = centroids[new_cluster_assigned]
    # If it's the first point within the cluster, then it completely determines the centroid
    new_weight = 1/cluster_sizes[new_cluster_assigned]
    centroid_new_loc = centroid_old_loc * (1-new_weight) + new_point * new_weight
    centroids[new_cluster_assigned] = centroid_new_loc
    return centroids

# Process one layout at a time
def clustering(layout, n_links_on):
    N = np.shape(layout)[0]
    assert np.shape(layout) == (N,4)
    # Take the middle point of each transceiver pair
    # These are the points for clustering
    layout_midpoints = np.stack(((layout[:,0]+layout[:,2])/2, (layout[:,1]+layout[:,3])/2), axis=1)
    assert np.shape(layout_midpoints) == (N,2)
    # Get the kmean++ centroids through calling sklearn KMeans with kmeans++ initialization
    # Without running clustering iterations, get k-means++ centroids from built in function
    km = KMeans(n_clusters=n_links_on, init='k-means++', n_init=10, max_iter=1).fit(layout_midpoints)
    centroids = km.cluster_centers_
    assert np.shape(centroids)==(n_links_on, 2)
    # Compute desired cluster sizes
    smaller_cluster_size = int(np.floor(N / n_links_on))
    residues = N - smaller_cluster_size*n_links_on
    cluster_size_targets = [smaller_cluster_size+1]*residues + [smaller_cluster_size]*(n_links_on-residues)
    assert np.sum(cluster_size_targets) == N
    # Fill in clusters based on: shortest distance to a cluster - longest distance to a cluster
    cluster_assignments = -np.ones(N, dtype=np.int8)
    cluster_sizes = np.zeros(n_links_on, dtype=np.int8)
    points_left = np.arange(N)
    clusters_left = np.arange(n_links_on)
    cluster_size_target = cluster_size_targets.pop(0)
    while np.size(points_left) > 0:
        # Have to recompute this since the centroids are moving
        points_to_centroids = np.reshape([np.linalg.norm(layout_midpoints[i]-centroids[j]) for i in points_left for j in clusters_left], [-1, np.size(clusters_left)])
        dists_spread = np.max(points_to_centroids, axis=1) - np.min(points_to_centroids, axis=1)
        point_to_assign_reduced_index = np.argmax(dists_spread)
        cluster_assigned_reduced_index = np.argmin(points_to_centroids[point_to_assign_reduced_index])
        # convert back to original full indexing
        point_to_assign_orig_index = points_left[point_to_assign_reduced_index]
        cluster_assigned_orig_index = clusters_left[cluster_assigned_reduced_index]
        cluster_assignments[point_to_assign_orig_index] = cluster_assigned_orig_index
        cluster_sizes[cluster_assigned_orig_index] += 1
        # update the centroid location with this additional point
        centroids = update_centroid(centroids, layout_midpoints[point_to_assign_orig_index], cluster_assigned_orig_index, cluster_sizes)
        # Remove this point as it's already assigned
        points_left = np.delete(points_left, point_to_assign_reduced_index)
        # Make a while loop checking since the cluster size target decreasing could lead to previously unsatisfied clusters now satisfied fulfillment
        while(np.max(cluster_sizes[clusters_left])==cluster_size_target):
            # Just fulfilled one cluster, since it's within the while loop, it doesn't have to be the cluster just fulfilled
            cluster_filled = np.argmax(cluster_sizes[clusters_left])
            clusters_left = np.delete(clusters_left, cluster_filled)
            if(np.size(cluster_size_targets)==0):
                # Should have fulfilled all the clusters
                assert np.size(points_left)==0
                break
            cluster_size_target = cluster_size_targets.pop(0)
    # Check whether indeed cluster sizes in the final cluster_assignments result are as desired
    assert np.shape(cluster_assignments)==(N,)
    assert np.all(np.logical_or(cluster_sizes == smaller_cluster_size, cluster_sizes == smaller_cluster_size+1))
    if (VISUALIZE):
        visualize_clusters.visualize_layout_clusters(layout, cluster_assignments, "K-Means Same Size")
    return cluster_assignments

def scheduling(layout, cluster_assignments):
    N = np.shape(layout)[0]
    assert np.shape(layout) == (N, 4)
    assert np.shape(cluster_assignments) == (N,)
    n_links_on = np.max(cluster_assignments) + 1
    allocations = np.zeros(N)
    # Select one shortest link from each cluster to schedule (since CSI isn't available)
    link_lengths = np.linalg.norm(layout[:, 0:2] - layout[:, 2:4], axis=1)
    for i in range(n_links_on):
        links_in_the_cluster = np.where(cluster_assignments == i)[0]
        shortest_link_in_the_cluster = links_in_the_cluster[np.argmin(link_lengths[links_in_the_cluster])]
        assert allocations[shortest_link_in_the_cluster] == 0, "having duplicate entry apperance across clusters"
        allocations[shortest_link_in_the_cluster] = 1
    return allocations