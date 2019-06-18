import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

ADJ_MAT_DESIGN = 'A'

SANITY_CHECK=True

def compute_spectral_clustering_explicit_kmeans(adj_mat, num_of_clusters):
    N = np.shape(adj_mat)[0]
    assert np.shape(adj_mat) == (N, N)
    assert np.transpose(adj_mat) == adj_mat
    laplace_mat = np.diag(np.sum(adj_mat, axis=1)) - adj_mat
    eig_vals, eig_vecs = np.linalg.eig(laplace_mat)
    smallest_eigval_indices = np.argsort(eig_vals)[:num_of_clusters]
    eig_vecs_selected = eig_vecs[:, smallest_eigval_indices]
    km = KMeans(n_clusters=num_of_clusters, n_init=100).fit(eig_vecs_selected)
    return km.labels_

def compute_spectral_clustering(adj_mat, num_of_clusters):
    N = np.shape(adj_mat)[0]
    assert np.shape(adj_mat) == (N, N)
    assert np.transpose(adj_mat) == adj_mat
    sc = SpectralClustering(n_clusters=num_of_clusters, affinity='precomputed', n_init=100, assign_labels='kmeans')
    assignments = sc.fit_predict(adj_mat)
    assert np.shape(assignments) == (N,)
    return assignments

# For now, just construct adjacency matrix layout by layout
def compute_adj_mat_from_CSI(CSI_mat):
    number_of_links = np.shape(CSI_mat)[0]
    assert np.shape(CSI_mat) == (number_of_links, number_of_links)
    if(ADJ_MAT_DESIGN=='A'):
        adj_mat = np.maximum(CSI_mat, np.transpose(CSI_mat)) # no need to clear-off the diagonal
    elif(ADJ_MAT_DESIGN=='B'):
        adj_mat = 0
    assert np.shape(adj_mat) == (number_of_links, number_of_links)
    return adj_mat

if(__name__=="__main__"):
    # check if calling the spectral clustering indeed returns results as the algorithm describes
    if(SANITY_CHECK):
        num_of_clusters = 5
        print("Sanity Check with Number of clusters: ", num_of_clusters)
        # Randomly generate adjacency matrix
        adj_mat = np.random.sample([7,7])
        adj_mat = np.maximum(adj_mat, np.transpose(adj_mat))
        # # Deterministic layouts (This layout illustrates my explicit calling method returning unreasonable solutions)
        # adj_mat = np.array([[0,1,1,0,0,0,0],
        #                     [1,0,1,1,0,1,0],
        #                     [1,1,0,0,0,0,0],
        #                     [0,1,0,0,1,0,0],
        #                     [0,0,0,1,0,0,0],
        #                     [0,1,0,0,0,0,1],
        #                     [0,0,0,0,0,1,0]])
        expcall_result = compute_spectral_clustering_explicit_kmeans(adj_mat, num_of_clusters)
        print("Explicitly call kmeans returns partitions: ")
        print(expcall_result)
        sc_result = compute_spectral_clustering(adj_mat, num_of_clusters)
        print("Direct SC object returns partitions: ")
        print(sc_result)
        print("Sanity Check Finished")
