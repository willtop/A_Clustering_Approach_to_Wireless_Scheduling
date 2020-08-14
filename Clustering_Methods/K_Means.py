# K-Means for clustering interferencing links
# To ensure the viability of using K-Means,gonna simply use the mid-point of each transceiver pair as the representative point of the link

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
sys.path.append("../Utilities/")
import utils

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
    assert np.shape(centroids) == (n_links_on, 2)
    if (VISUALIZE):
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("K-Means {} Clusters Visualization".format(n_links_on))
        utils.plot_stations_on_layout(ax, layout)
        utils.plot_clusters_on_layout(ax, layout, cluster_assignments)
        plt.tight_layout()
        plt.show()
    return cluster_assignments, centroids
