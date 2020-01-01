# K-Means for clustering interferencing links
# To ensure the viability of using K-Means, with the gauranteed stabelizing clustering
# Gonna simply use the mid-point of each transceiver pair as the representative point of the link

import numpy as np
from sklearn.cluster import KMeans
import visualize_clusters

VISUALIZE = False
# Process one layout at a time
def clustering(layout, n_links_on):
    N = np.shape(layout)[0]
    assert np.shape(layout) == (N,4)
    # Take the middle point of each transceiver pair
    layout_midpoints = np.stack(((layout[:,0]+layout[:,2])/2, (layout[:,1]+layout[:,3])/2), axis=1)
    assert np.shape(layout_midpoints) == (N,2)
    km = KMeans(n_clusters=n_links_on, n_init=10).fit(layout_midpoints)
    cluster_assignments = km.labels_
    centroids = km.cluster_centers_
    assert np.shape(cluster_assignments)==(N,)
    assert np.shape(centroids)==(n_links_on, 2)
    if (VISUALIZE):
        visualize_clusters.visualize_layout_clusters(layout, cluster_assignments, centroids)
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