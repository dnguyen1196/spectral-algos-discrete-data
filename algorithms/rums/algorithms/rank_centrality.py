import numpy as np
from scipy.special import softmax
import time

# How should we handle the situation when some items have no outflow -> should we just truncate those at 1./n**2? What's the rationale for this?

class RankCentrality():
    def __init__(self, n, nu=1):
        self.n = n
        self.nu = nu
        self.iter_counts = None

    def fit(self, ranked_data, max_iters_mc=10000, eps_mc=1e-6, theta_init=None, verbose=False):

        start = time.time()
        if verbose:
            print("Choice breaking ... ")
        comparisons = self.rank_break(ranked_data)
        
        if verbose:
            print(f"Choice breaking took {time.time() - start} seconds")

        if theta_init is None:
            theta_init = np.zeros((self.n,))
        theta = theta_init
        
        pi = softmax(theta)[:, np.newaxis]
        pi = pi.T
        assert(pi.shape == (1, self.n))

        start = time.time()
        M, d = self.construct_markov_chain(comparisons)
        mc_construction_time = time.time() - start

        # Compute stationary distribution
        start = time.time()
        pi_, iter_counts = self.compute_stationary_distribution(M, pi, max_iters_mc, eps_mc, return_iter_count=True)
        mc_convergence_time = time.time() - start
        self.iter_counts = iter_counts

        if verbose:
            print(f"The MC took {mc_construction_time} to construct and {mc_convergence_time} to converge after {iter_counts} iterations")

        # Normalize
        pi_[0, :] = pi_[0, :]/d
        pi_[0, :] = pi_[0, :] / np.sum(pi_)

        # Estimate item parameters
        theta = np.log(pi_.flatten())
        theta -= np.mean(theta)

        self.theta = np.copy(theta)
        return self.theta
    
    
    def fit_from_pairwise_probabilities(self, P, max_iters_mc=10000, lambd_mc=0., eps_mc=1e-8, theta_init=None, verbose=False):
        M = np.where(np.logical_or((P != 0), (P.T != 0)), P+lambd_mc, P) # Add a very small regularization
        np.fill_diagonal(M, 0)
        d_max = np.max(np.sum(M,1))
        d = np.ones((self.n,)) * d_max
        # d = np.sum(M, 1)
        for i in range(self.n):
            M[i, :] /= d[i]
            M[i, i] = 1. - np.sum(M[i, :])
        
        if theta_init is None:
            theta_init = np.zeros((self.n,))
        theta = theta_init
        pi = softmax(theta)[:, np.newaxis]
        pi = pi.T
        assert(pi.shape == (1, self.n))
        
        # Compute stationary distribution
        start = time.time()
        pi_, iter_counts = self.compute_stationary_distribution(M, pi, max_iters_mc, eps_mc, return_iter_count=True)
        mc_convergence_time = time.time() - start
        self.iter_counts = iter_counts

        # Normalize
        pi_[0, :] = pi_[0, :]/d
        pi_[0, :] = pi_[0, :] / np.sum(pi_)

        # Estimate item parameters
        theta = np.log(pi_.flatten())
        theta -= np.mean(theta)

        self.theta = np.copy(theta)
        return self.theta
    
    def rank_break(self, rankings):
        comparisons = []
        for rank in rankings:
            for idi, i in enumerate(rank[:-1]):
                for j in rank[idi+1:]:
                    comparisons.append((i, j))

        return comparisons
    
    def construct_markov_chain(self, comparisons):
        M = np.zeros((self.n, self.n))
        for winner, loser in comparisons:
            M[loser, winner] += 1
        
        # Check everypair where if i flows into j, j should also have back flow 
        M = np.where(np.logical_or((M != 0), (M.T != 0)), M+self.nu, M)
        
        for i in range(self.n-1):
            for j in range(i+1, self.n):
                if M[i, j] != 0 or M[j, i] != 0:
                    L = M[i, j] + M[j, i]
                    M[i, j] = M[i, j] / L
                    M[j, i] = M[j, i] / L
                    
        # d = np.count_nonzero(M, 1)
        d_max = np.max(np.sum(M,1))
        d = np.ones((self.n,)) * d_max

        # d = np.sum(M, 1)
        for i in range(self.n):
            M[i, :] /= d[i]
            M[i, i] = 1. - np.sum(M[i, :])

        return M, d

    def compute_stationary_distribution(self, M, init_pi=None, max_iters=10000, eps=1e-6, return_iter_count=False):
        if init_pi is None:
            pi = np.ones((1,self.n)) * 1./self.n
        else:
            pi = init_pi
        
        for it in range(max_iters):
            pi_ = pi @ M
            if np.linalg.norm(pi_ - pi) < eps:
                break
            pi = pi_
        if return_iter_count:
            return pi, it
        return pi