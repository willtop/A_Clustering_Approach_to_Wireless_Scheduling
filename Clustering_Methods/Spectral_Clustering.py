# For any reproduce, further research or development, please kindly cite our SPAWC conference paper: 
# @Conference{cluster_schedule, 
#             author = "W. Cui and W. Yu", 
#             title = "A Clustering Approach to Wireless Scheduling", 
#             booktitle = "IEEE Workshop Signal Process. Advances Wireless Commun. (SPAWC)", 
#             year = 2020, 
#             month = may }
#

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
sys.path.append("../Utilities/")
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
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Spectral Clustering {} Clusters Visualization".format(n_links_on))
        utils.plot_stations_on_layout(ax, layout)
        utils.plot_clusters_on_layout(ax, layout, cluster_assignments)
        plt.tight_layout()
        plt.show()
    return cluster_assignments