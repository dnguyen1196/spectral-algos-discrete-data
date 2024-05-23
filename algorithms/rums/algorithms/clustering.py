from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
from scipy.linalg import svd
from .embeddings import PairwiseEmbedding
from sklearn.decomposition import PCA
import cvxpy as cp
from sklearn import random_projection
import scipy.sparse as ss


class SpectralClustering:
    def __init__(self, n, K, embedding=None, est="PCA", lambd=1, clustering_algo="k-means"):
        self.n = n
        self.K = K
        if embedding is None:
            embedding = PairwiseEmbedding(n)
        self.embedding = embedding
        self.labels = None
        assert(est in ["PCA", "Nuc", "Alt", "PCA-mod", "rand-gauss", "rand-sparse", None])
        self.est = est
        self.lambd = lambd
        assert(clustering_algo in ["k-means", "agglomerative"])
        self.clustering_algo = clustering_algo

    def fit(self, ranked_data, is_embedded=False):
        # Embed ranked data
        if not is_embedded:
            X = self.embedding.fit(ranked_data)
        else:
            # If the data is aleady embedded
            X = ranked_data

        # Run a low rank approximation algorithm
        Y = self.low_rank_approximate(X)

        # Then run Agglomerative Clustering
        if self.clustering_algo == "k-means":
            clustering = KMeans(self.K)
        elif self.clustering_algo == "agglomerative":
            clustering = AgglomerativeClustering(self.K)
        
        clustering.fit(Y)
        self.labels = clustering.labels_
        return self.labels, clustering.cluster_centers_

    def low_rank_approximate(self, X):
        if self.est == "PCA":
            return self.estimate_pca(X)
        elif self.est == "Nuc":
            return self.estimate_matrix_completion(X)
        elif self.est == "Alt":
            return self.estimate_alternating_minimization(X)
        elif self.est == "PCA-mod":
            return self.estimate_pca_modified(X)
        elif self.est == "rand-gauss":
            return self.estimate_gaussian_randomized_projection(X)
        elif self.est == "rand-sparse":
            return self.estimate_sparse_random_projection(X)
        elif self.est is None:
            return X
        else:
            raise NotImplementedError
    
    def estimate_pca(self, X):
        # Project onto the principal subspace
        # if np.count_nonzero(X) < np.prod(X.shape): # Sparse
        X_sparse = ss.coo_matrix(X)
        Y, _, _ = ss.linalg.svds(X_sparse, self.K)
        # else:
        #     Y, _, _ = svd(X)
        return Y

    def estimate_pca_modified(self, X):
        # Project onto the SCALED principal subspace
        # if np.count_nonzero(X) < np.prod(X.shape): # Sparse
        X_sparse = ss.coo_matrix(X)
        U, s, _ = ss.linalg.svds(X_sparse, self.K)
        Y = U @ np.diag(s)
        # else:
        #     U, s, _ = svd(X)
        #     Y = U[:, :self.K] @ np.diag(s[:self.K])
        return Y

    def estimate_gaussian_randomized_projection(self, X):
        m = len(X)
        transformer = random_projection.GaussianRandomProjection(int(8*np.log(m)))
        return transformer.fit_transform(X)
    
    def estimate_sparse_random_projection(self, X):
        m = len(X)
        transformer = random_projection.SparseRandomProjection(int(8*np.log(m)))
        return transformer.fit_transform(X)

    def estimate_matrix_completion(self, X):
        N, d = X.shape
        M = cp.Variable((N, d))

        loss = 1./2 * cp.norm(X - M, "fro")**2 + self.lambd * cp.norm(M, "nuc")
        objective = cp.Minimize(loss)
        problem = cp.Problem(objective)
        problem.solve()
        return M.value

    def estimate_alternating_minimization(self, X, max_iters=100, tol=1e-3):
        # Then perform alternating least squares minimization until convergence
        def minimize_V(X, U):
            d, K = V.shape
            V_ = cp.Variable((d, K))
            loss = 1./2 * cp.norm(X - U @ V_.T, "fro") ** 2
            problem = cp.Problem(cp.Minimize(loss))
            problem.solve()
            return V_.value

        def minimize_U(X, V):
            N, K = U.shape
            U_ = cp.Variable((N, K))
            loss = 1./2 * cp.norm(X - U_ @ V.T, "fro") ** 2
            problem = cp.Problem(cp.Minimize(loss))
            problem.solve()
            return U_.value

        # Initialize with SVD
        U, S, Vh = np.linalg.svd(X)
        U_0 = U[:, :self.K] @ np.diag(np.sqrt(S[:self.K]))
        V_0 = (Vh[:self.K, :]).T @ np.diag(np.sqrt(S[:self.K]))

        U = U_0
        V = V_0
        for it in range(max_iters):
            U_next = minimize_U(X, V)
            V_next = minimize_V(X, U_next)
            if np.sum(np.square(U_next - U)) < tol and np.sum(np.square(V_next - V)) < tol:
                U = U_next
                V = V_next
                break
            U = U_next
            V = V_next

        return U @ V.T