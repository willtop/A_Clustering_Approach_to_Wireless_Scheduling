import numpy as np
from sklearn.cluster import KMeans
from scipy import linalg as LA
import matplotlib.pyplot as plt
import sys
sys.path.append("../Tools/")
import utils

ADJ_MAT_DESIGN = 'A'
VISUALIZE = True

# For now, just construct adjacency matrix layout by layout
def compute_adj_mat_from_gains(gains_mat):
    number_of_links = np.shape(gains_mat)[0]
    assert np.shape(gains_mat) == (number_of_links, number_of_links)
    if(ADJ_MAT_DESIGN=='A'):
        adj_mat = np.maximum(gains_mat, np.transpose(gains_mat)) # no need to clear-off the diagonal
    elif(ADJ_MAT_DESIGN=='B'):
        adj_mat = 0
    assert np.shape(adj_mat) == (number_of_links, number_of_links)
    return adj_mat

def visualize_layout_clusters(layout, cluster_assignments):
    num_of_links = np.shape(layout)[0]
    num_of_clusters = np.max(cluster_assignments)+1 # zero indexing
    assert np.shape(layout)==(num_of_links, 4)
    assert np.shape(cluster_assignments)==(num_of_links, )
    # assign colors to clusters
    color_map = plt.get_cmap('gist_rainbow', num_of_clusters)
    links_colors = []
    for i in range(num_of_links):
        links_colors.append(color_map(cluster_assignments[i]))
    utils.plot_allocs_on_layout(plt.gca(), layout, links_colors, "Clusters Visualization with {} clusters".format(num_of_clusters), whether_allocs=False, cluster_assignments=cluster_assignments)
    plt.show()
    return


# For now, schedule one layout at a time
def schedule(general_para, layout, gains_mat):
    N = np.shape(gains_mat)[0]
    assert np.shape(gains_mat) == (N, N)
    assert np.shape(layout) == (N, 4)
    adj_mat = compute_adj_mat_from_gains(gains_mat)
    assert np.shape(adj_mat) == (N, N)
    assert np.all(np.transpose(adj_mat) == adj_mat)
    laplace_mat = np.diag(np.sum(adj_mat, axis=1)) - adj_mat
    eig_vals, eig_vecs = np.linalg.eig(laplace_mat)
    if (VISUALIZE):
        sorted_eig_vals = np.sort(eig_vals)
        eig_vals_diff = np.diff(sorted_eig_vals)
        eig_vals_diff_ratios = eig_vals_diff[:-1] / eig_vals_diff[1:]
        turning_eig_val_index = np.argmin(eig_vals_diff_ratios) + 1
        plt.title("Eigenvalues of {}X{} Laplacian Matrix".format(N,N))
        plt.plot(sorted_eig_vals, "*-")
        for i in range(np.size(eig_vals)):
            plt.annotate(i + 1, [i, sorted_eig_vals[i]])
        plt.show()
    # Try number of clusters one by one
    max_sumrate = 0
    gains_diagonal = utils.get_diagonal_gains(np.expand_dims(gains_mat,axis=0))
    gains_nondiagonal = utils.get_nondiagonal_gains(np.expand_dims(gains_mat,axis=0))
    for num_of_links_on in range(2, N): # not including activating just one link or all active
        smallest_eigval_indices = np.argsort(eig_vals)[1:num_of_links_on] # no need to take the zero eigen value
        eig_vecs_selected = eig_vecs[:, smallest_eigval_indices]
        km = KMeans(n_clusters=num_of_links_on, n_init=10).fit(eig_vecs_selected)
        assignments = km.labels_
        if (VISUALIZE):
            visualize_layout_clusters(layout, assignments)
        allocations = np.zeros(N)
        # Select one strongest link from each cluster to schedule
        for i in range(num_of_links_on):
            links_in_the_cluster = np.where(assignments == i)[0]
            strongest_link_in_the_cluster = links_in_the_cluster[np.argmax(np.diag(gains_mat)[links_in_the_cluster])]
            assert allocations[strongest_link_in_the_cluster] == 0, "having duplicate entry apperance across clusters"
            allocations[strongest_link_in_the_cluster] = 1
        # compute the corresponding rate to this allocation
        rates = utils.compute_rates(general_para, np.expand_dims(allocations, axis=0), gains_diagonal, gains_nondiagonal)
        assert np.shape(rates)==(1,N)
        sumrate = np.sum(rates)
        if(sumrate > max_sumrate):
            best_allocations = allocations
            max_sumrate = sumrate
    return best_allocations
