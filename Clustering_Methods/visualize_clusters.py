import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../Utilities_Research/")
import utils

def visualize_layout_clusters(layout, cluster_assignments, centroids=[]):
    N = np.shape(layout)[0]
    n_clusters = np.max(cluster_assignments)+1 # zero indexing
    assert np.shape(layout)==(N, 4)
    assert np.shape(cluster_assignments)==(N, )
    if np.size(centroids)>0:
        assert np.shape(centroids)==(n_clusters, 2)
    # assign colors to clusters
    color_map = plt.get_cmap('gist_rainbow', n_clusters)
    links_colors = []
    for i in range(N):
        links_colors.append(color_map(cluster_assignments[i]))
    ax = plt.gca()
    utils.plot_allocs_on_layout(ax, layout, links_colors, "Clusters Visualization with {} clusters".format(n_clusters), whether_allocs=False, cluster_assignments=cluster_assignments)
    ax.scatter(centroids[:,0], centroids[:,1], marker="*", label="Centroids", s=11)
    plt.show()
    return