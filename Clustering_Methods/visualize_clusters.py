import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../Utilities_Research/")
import utils

def visualize_layout_clusters(layout, cluster_assignments, method_name):
    N = np.shape(layout)[0]
    n_clusters = np.max(cluster_assignments)+1 # zero indexing
    assert np.shape(layout)==(N, 4)
    assert np.shape(cluster_assignments)==(N, )
    # assign colors to clusters
    color_map = plt.get_cmap('gist_rainbow', n_clusters)
    links_colors = []
    for i in range(N):
        links_colors.append(color_map(cluster_assignments[i]))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    utils.plot_allocs_on_layout(ax, layout, links_colors, "{} Clusters Visualization".format(method_name), whether_allocs=False, cluster_assignments=cluster_assignments)
    plt.tight_layout()
    plt.show()
    return