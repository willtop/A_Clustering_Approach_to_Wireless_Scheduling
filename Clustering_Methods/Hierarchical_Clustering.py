# For any reproduce, further research or development, please kindly cite our SPAWC conference paper: 
# @Conference{cluster_schedule, 
#             author = "W. Cui and W. Yu", 
#             title = "A Clustering Approach to Wireless Scheduling", 
#             booktitle = "IEEE Workshop Signal Process. Advances Wireless Commun. (SPAWC)", 
#             year = 2020, 
#             month = may }
#
# Hierarchical Clustering taking the adjacency matrix (CSI) inputs

import numpy as np
from sklearn.cluster import AgglomerativeClustering
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
    return -adj_mat # return negative channel channel_losses so stronger channels correspond to less values (shorter distance)

# For now, process one layout at a time
def clustering(layout, channel_losses_mat, n_links_on):
    N = np.shape(channel_losses_mat)[0]
    assert np.shape(channel_losses_mat) == (N, N)
    assert np.shape(layout) == (N, 4)
    adj_mat = construct_adj_mat(channel_losses_mat)
    assert np.shape(adj_mat) == (N, N)
    cluster_assignments = AgglomerativeClustering(n_clusters=n_links_on, affinity='precomputed', linkage='average').fit_predict(adj_mat)
    assert np.shape(cluster_assignments) == (N, )
    if (VISUALIZE):
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Hierarchical Clustering {} Clusters Visualization".format(n_links_on))
        utils.plot_stations_on_layout(ax, layout)
        utils.plot_clusters_on_layout(ax, layout, cluster_assignments)
        plt.tight_layout()
        plt.show()
    return cluster_assignments