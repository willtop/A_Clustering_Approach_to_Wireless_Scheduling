import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import scipy

ADJ_MAT_DESIGN = 'A'

SANITY_CHECK=True

def laplacian(A):
    """Computes the symetric normalized laplacian.
    L = D^{-1/2} A D{-1/2}
    """
    D = np.zeros(A.shape)
    w = np.sum(A, axis=0)
    D.flat[::len(w) + 1] = w ** (-0.5)  # set the diag of D to w
    return D.dot(A).dot(D)


def k_means(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=1231)
    return kmeans.fit(X).labels_


def spectral_clustering_3rd(affinity, n_clusters, cluster_method=k_means):
    L = laplacian(affinity)
    eig_val, eig_vect = scipy.sparse.linalg.eigs(L, n_clusters)
    X = eig_vect.real
    rows_norm = np.linalg.norm(X, axis=1, ord=2)
    Y = (X.T / rows_norm).T
    labels = cluster_method(Y, n_clusters)
    return labels

def compute_spectral_clustering_explicit_kmeans(adj_mat, num_of_clusters):
    N = np.shape(adj_mat)[0]
    assert np.shape(adj_mat) == (N, N)
    assert np.all(np.transpose(adj_mat) == adj_mat)
    laplace_mat = np.diag(np.sum(adj_mat, axis=1)) - adj_mat
    eig_vals, eig_vecs = np.linalg.eig(laplace_mat)
    smallest_eigval_indices = np.argsort(eig_vals)[:num_of_clusters]
    eig_vecs_selected = eig_vecs[:, smallest_eigval_indices]
    km = KMeans(n_clusters=num_of_clusters, n_init=100).fit(eig_vecs_selected)
    return km.labels_

def compute_spectral_clustering(adj_mat, num_of_clusters):
    N = np.shape(adj_mat)[0]
    assert np.shape(adj_mat) == (N, N)
    assert np.all(np.transpose(adj_mat) == adj_mat)
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
        num_of_clusters = 3
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
        # try a 3rd online explictly coded different method 
        third_result = spectral_clustering_3rd(adj_mat, num_of_clusters)
        print("3rd online implementation: ")
        print(third_result)
        print("Sanity Check Finished")


