import numpy as np
import scipy as sp
import time
from scipy.sparse import csr_matrix


##############################################################################################################
#
#
#                       The spectral algorithm for the polytomous Rasch model
#
#
##############################################################################################################    
    
class SpectralAlgorithm:
    def __init__(self, lambd=0.1):
        """
        Args:
            lambd: float, regularization parameter to avoid numerical issues when computing the stationary distribution
        
        """
        self.precomputed_Y = {} # This will be used to save computations
        self.lambd = lambd
    
    def fit(self, X, sample_weight=None):
        """
        Args:
            X: Input response matrix of shape (n_users, n_itmes) 
            sample_weight: optional vector of per user sample weight
        
        
        L is the max grade level (0, 1, 2, ..., L)
        
        1. Obtain shifted estimate for each level l = 1, ..., L
            by computing the stationary distribution of the Markov chain constructed from
            Y(l-1, l)
            
            Denote these as tilde tilde theta^1, ..., tilde theta^L
        
        2. Normalize level 1 parameter to have mean 0 -> Denote as hat theta^1
        
        3. For each level l = 2,...,L, estimate the shift delta_l by
            
            sum_{j=1}^m sum_{i not j} e^{hat theta_i^{l-1}} * Y_ij(l-1,l-1)
            
            DIVIDED by
            
            sum_{j=1}^m e^{tilde theta_j^{l}} sum_{i not j} Y_{ij}(l, l-2)
            
            Recover the level l parameter estimate hat theta^l = tilde theta^l + delta_l
        
        ---------------
        Returns
            m x K parameters for the tests' difficulties
        
        """
        if sample_weight is None:
            sample_weight = np.ones((X.shape[0], ))
            
        L = int(np.max(X))
        normalized_betas = self.compute_normalized_betas(X, L, sample_weight)
        unnormalized_betas = self.compute_unnormalized_betas(X, normalized_betas, L, sample_weight)
        return unnormalized_betas
    
    def compute_normalized_betas(self, X, L, sample_weight=None):
        """Perform the first phase of the spectral algorithm to obtain level-wise normalized estimate

        Args:
            X (matrix of shape (n_users, n_items)): Input data
            L (int): Number of levels
            sample_weight (float vector, optional): User sample weight. Defaults to None.
            verbose (bool, optional): Defaults to False.

        Returns:
            Normalized betas
        """
        normalized_betas = []
        for l in range(1, L+1): # For l = 1,2,...,L
            Y_ll1 = self.construct_Y_matrix(X, l, l-1, sample_weight)
            M_ll1, d = self.construct_markov_chain(Y_ll1)
            pi = self.compute_stationary_distribution(M_ll1)
            pi = pi.flatten()
            pi = pi/d
            normalized_betas.append(np.log(pi))
        return np.array(normalized_betas)
    
    
    def compute_unnormalized_betas(self, X, normalized_betas, L, sample_weight=None):
        """

        Args:
            X: input data matrix
            normalized_betas: output of the first phase of the spectral algorithm
            L: number of levels, need to match with normalized_betas.shape[1]
            sample_weight: user-wise sample weight
        
        Returns
            The final beta estimate
        
        """
                
        unnormalized_betas = []
        shifted_betas_1 = normalized_betas[0, :] - np.mean(normalized_betas[0, :])
        unnormalized_betas.append(shifted_betas_1)
        n, m = X.shape
        
        for k in range(2, L+1):
            Y_1_km1 = self.construct_Y_matrix(X, 1, k-1, sample_weight)
            Y_0_k = self.construct_Y_matrix(X, 0, k, sample_weight)
            np.fill_diagonal(Y_1_km1, 0)
            np.fill_diagonal(Y_0_k, 0)
            
            shifted_betas_k = normalized_betas[k-1, :] - np.mean(normalized_betas[k-1,:])
            numer = Y_1_km1 * (np.exp(shifted_betas_1)).reshape((-1, 1))
            denom = Y_0_k * (np.exp(shifted_betas_k)).reshape((1, -1))
            
            delta_k = np.log(np.sum(numer)/np.sum(denom))            
            unnormalized_betas.append(shifted_betas_k + delta_k)
        
        return np.array(unnormalized_betas).T
    
        
    def construct_markov_chain(self, Y):
        """
        Given Y matrix, construct a degree-unnormalized Markov chain
        
        """
        M = np.copy(Y).astype(float)
        M = np.where(np.logical_or((M != 0), (M.T != 0)), M + self.lambd, M)
        
        m = M.shape[0]
        d = np.maximum(np.sum(M, 1), 1)
        for i in range(m):
            M[i, :] = M[i, :] / d[i]
            M[i, i] = 1. - np.sum(M[i, :])
        return M, d
    
    def compute_stationary_distribution(self, M, pi_init=None, max_iters=10000, eps=1e-6):
        """
        Compute the stationary distribution of a Markov chain
        
        """
        m = M.shape[0]
        if pi_init is None:
            pi_init = np.ones((m, )).T
        pi = pi_init
        for _ in range(max_iters):
            pi_next = (pi @ M)
            pi_next /= np.sum(pi_next)
            if np.linalg.norm(pi_next - pi) < eps:
                pi = pi_next
                break
            pi = pi_next
        return pi
    
    def construct_Y_matrix(self, X, k, k_prime, sample_weight=None):
        """Construct the Y matrix used in both phases of the spectral algorithm

        Args
            X: data
            k, k_prime: two levels needed to compute Y_{ij}^{k, k_prime}
            sample_weight: user-wise sample weight
            
        Returns
            Y matrix of shape (n_items, n_items)
        
        """
        # Note: this is the only function that directly interacts with the data X
        # It automatically handles missing data
        # assert(k <= k_prime)
        # For every pair i, j
        # Yij = #{student with score k for i and score k' for j}
        # If k' is higher -> then j is 'easier'
        # Then the Markov chain should flip this direction, since we want the parameter for j 
        # to be lower. 
        
        if sample_weight is None:
            sample_weight = np.ones((X.shape[0], 1))
        
        if (k, k_prime) in self.precomputed_Y:
            return self.precomputed_Y[(k, k_prime)]
        is_k = csr_matrix((X == k).astype(int) * sample_weight.reshape((-1, 1)))
        is_k_prime = csr_matrix((X == k_prime).astype(int) * sample_weight.reshape((-1, 1)))
        # Yij = sum_n is_k[n, i] * is_k'[n, j]
        #     = np.sum(is_k[:, i] * is_k'[n, j])
        #     = (is_k.T @ is_k')_{ij}
        Y = (is_k.T @ is_k_prime).toarray()
        self.precomputed_Y[(k, k_prime)] = Y
        return Y

