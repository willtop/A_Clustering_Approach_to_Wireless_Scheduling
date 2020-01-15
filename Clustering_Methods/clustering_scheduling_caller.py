# This is a script that serves as a caller invoking different clustering scripts based on a string parameter
import numpy as np
import Spectral_Clustering
import Hierarchical_Clustering
import K_Means
import Hierarchical_Clustering_EqualSize
import K_Means_EqualSize

def clustering(layout, channel_losses_mat, n_links_on, method_name):
    N = np.shape(layout)[0]
    assert np.shape(layout) == (N, 4)
    assert np.shape(channel_losses_mat) == (N, N)
    assert 0 <= n_links_on <= N
    if (method_name == "Spectral Clusterinig"):
        clusters_one_layout = Spectral_Clustering.clustering(layout, channel_losses_mat, n_links_on)
    elif (method_name == "Hierarchical Clustering"):
        clusters_one_layout = Hierarchical_Clustering.clustering(layout, channel_losses_mat, n_links_on)
    elif (method_name == "K-Means"):
        clusters_one_layout = K_Means.clustering(layout, channel_losses_mat, n_links_on)
    elif (method_name == "Hierarchical Clustering EqualSize"):
        clusters_one_layout = Hierarchical_Clustering_EqualSize.clustering(layout, channel_losses_mat, n_links_on)
    elif (method_name == "K-Means EqualSize"):
        clusters_one_layout = K_Means_EqualSize.clustering(layout, channel_losses_mat, n_links_on)
    else:
        print("Invalid clustering method name: {}!".format(method_name))
        exit(1)
    return clusters_one_layout

def scheduling(inputs, cluster_assignments, method_name):
    N = np.shape(inputs)[0]
    # Could be either GLI (for K-Means) or CSI(for other clustering methods)
    assert np.shape(inputs) == (N, 4) or np.shape(inputs) == (N, N)
    assert np.shape(cluster_assignments) == (N, )
    if (method_name == "Spectral Clusterinig"):
        allocs_one_layout = Spectral_Clustering.scheduling(inputs, cluster_assignments)
    elif (method_name == "Hierarchical Clustering"):
        allocs_one_layout = Hierarchical_Clustering.scheduling(inputs, cluster_assignments)
    elif (method_name == "K-Means"):
        allocs_one_layout = K_Means.scheduling(inputs, cluster_assignments)
    elif (method_name == "Hierarchical Clustering EqualSize"):
        allocs_one_layout = Hierarchical_Clustering_EqualSize.clustering(inputs, cluster_assignments)
    elif (method_name == "K-Means EqualSize"):
        allocs_one_layout = K_Means_EqualSize.scheduling(inputs, cluster_assignments)
    else:
        print("Invalid clustering method name: {}!".format(method_name))
        exit(1)
    return allocs_one_layout
