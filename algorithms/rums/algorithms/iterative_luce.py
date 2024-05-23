import numpy as np
from scipy.special import softmax
import time

class RegularizedILSR():
    def __init__(self, n, a0=None, b0=None, mle=False, nu=0.01):
        """
        Generalized Luce spectral ranking algorithm. It can perform MLE likelihood for
        Luce models (when mle is set to True) or Bayesian (when mle is set to False
        and a0, b0 are given). To learn more about Bayesian inference under the Plackett
        Luce models, consult [1].
        
        [1] Caron and Doucet https://arxiv.org/pdf/1011.1761.pdf

        :param n: Number of items
        :type n: int
        :param a0: prior parameter, defaults to None
        :type a0: np.array, optional
        :param b0: prior parameter, defaults to None
        :type b0: np.array, optional
        :param mle: flag whether we want to do MLE or Bayesian, defaults to False
        :type mle: bool, optional
        :param nu: small regularization parameter for MLE to get numerical stability, defaults to 0.01
        :type nu: float, optional
        """
        self.n = n
        if a0 is None:
            a0 = np.ones((n,))
        self.a0 = a0
        if b0 is None:
            b0 = np.ones((n,)) * n
        self.b0 = b0
        self.mle = mle
        self.nu = 0.01
        
    def construct_choice_tensor(self, ranked_data):
        """ Given (partial) rankings data, construct a data structure used in Markov chain construction

        :param ranked_data: Rankings data of size m
        :type ranked_data: list of list
        """
        # This should return a (sparse) tensor of size (n, n, m)
        # S where S[j, i, :] = {0,1}^l where S[j, i, l] = 1 if j is ranked ahead of i in pi_l        
        # and a sparse matrix not_last of size (n, m) where not_last[i, l] = 1 if item i is NOT ranked last in ranking l
        winning_idx_S = []
        losing_idx_S = []
        sample_idx_S = []
        
        item_idx_not_last = []
        sample_idx_not_last = []
        
        for l, rank in enumerate(ranked_data):
            for idj, j in enumerate(rank[:-1]): # Everything except for deadlast item
                item_idx_not_last.append(j)
                sample_idx_not_last.append(l)
                    
                for i in rank[idj+1:]: # Mark j beats i
                    winning_idx_S.append(j)
                    losing_idx_S.append(i)
                    sample_idx_S.append(l)
                    
        # These will be sparse eventually
        S = pysparse.COO(
            [ winning_idx_S, losing_idx_S, sample_idx_S],
            np.ones((len(sample_idx_S,))),
            shape=(self.n, self.n, len(ranked_data))
        )
        not_last = csr_matrix(
            (np.ones((len(sample_idx_not_last),)), (item_idx_not_last, sample_idx_not_last)), shape=(self.n, len(ranked_data)) 
        )
        return S, not_last
    
    def construct_markov_chain_accelerated(self, S_choice_tensor, pi_augmented, not_last):
        if self.mle:
            pi = pi_augmented
        else:
            pi = pi_augmented[:-1]
        temp = pi @ S_choice_tensor + not_last.multiply(pi[:, np.newaxis])
        # temp has shape (n, m) 
        # temp[j, :] = pi @ S_choice_tensor[j, :, :] + pi[j] * indic(j is NOT dead last in ranking l)
        piSk = np.divide(1, temp, out=np.zeros_like(temp), where=temp!=0)
        # piSk has shape (n, m) where piSk[j, l] = 1./(sum_{k\in Sl} pi_k) , 
        # essentially the weight of all items involved in the choice break
        # M_sub = (np.transpose(S_choice_tensor, (1, 0, 2)) * piSk).sum(-1)
        M_sub = (S_choice_tensor.transpose((1,0,2)) * piSk).sum(-1).todense()        
        np.fill_diagonal(M_sub, 0)
        
        if not self.mle:
            M = np.zeros((self.n+1, self.n+1))
            # Fill out the remaining entries of M
            M[:-1, :-1] = M_sub
            M[:-1, self.n] = self.b0
            M[self.n, :-1] = self.a0
        else:
            M = np.where(np.logical_and((M_sub == 0), (M_sub.T != 0)), M_sub+self.nu, M_sub)
            
        d = np.sum(M, 1)
        for i in range(M.shape[0]):
            M[i, :] /= d[i]
            
        return csc_matrix(M), d
        
    def fit(self, ranked_data, max_iters=100, tol=1e-3, verbose=False, mc_method="eigval"):
        """ Fit on partial rankings data

        :param ranked_data: rankings data
        :type ranked_data: list of list
        :param max_iters: max number of iterations, defaults to 100
        :type max_iters: int, optional
        :param tol: Tolerance for convergence, defaults to 1e-3
        :type tol: float, optional
        :param verbose: flag whether to print out time statistics, defaults to False
        :type verbose: bool, optional
        :param mc_method: method to compute the stationary distribution of Markov chain, defaults to "eigval"
            either eigval which computes the leading eigenvector using sparse linear algebra or power method
            which repeatedly applies pi @ M to compute the stationary distribution of Markov chain M.
        :type mc_method: str, optional
        
        
        
        :return: If MLE is True, returns lambd, it where 
            lambd is the item parameter estimate 
            `it' is the number of iterations run
            
            If MLE is False, returns lambd, dummy, it
                lambd, it are explained above.
                dummy is the parameter estimate for the "dummy" state. If one only
                cares about MAP estiamte, this is not needed. This is needed if one
                wants to sample from the posterior distribution
         
        """
        assert(mc_method in ["eigval", "power"])
        start = time.time()
        choice_tensor, not_last = self.construct_choice_tensor(ranked_data)

        if verbose:
            print("Constructing choice tensor ", time.time() - start)
        
        if self.mle:
            lambd_augmented = np.ones((self.n,)) / self.n
        else:
            lambd_augmented = np.ones((self.n+1,))/(self.n+1) # Create a dummy item
        lambd_augmented = lambd_augmented[:, np.newaxis].T
        
        for it in range(max_iters):
            start = time.time()
            M, d = self.construct_markov_chain_accelerated(choice_tensor, lambd_augmented.flatten(), not_last)
            mc_construct_time = time.time() - start
            
            lambd_init = lambd_augmented * d

            lambd_init[0,:] = lambd_init[0,:] / np.sum(lambd_init)
            start = time.time()
            if mc_method == "power":
                lambd_next, it_ssd = self.compute_stationary_distribution(M, lambd_init, self.max_mc_iters)
            else:
                lambd_next, it_ssd = self.compute_stationary_distribution_linalg(M), 0
            
            ssd_compute_time = time.time() - start
            
            if verbose:
                print(f"Iter {it}, MC construction: {mc_construct_time}, " + 
                      f"SSD compute: {ssd_compute_time} ({it_ssd})")
            
            lambd_next /= d
            lambd_next[0,:] = lambd_next[0,:] / np.sum(lambd_next)
            
            if np.linalg.norm(lambd_next - lambd_augmented) < tol:
                break
            lambd_augmented = lambd_next

        lambd = lambd_augmented.flatten()
        if self.mle:
            return lambd, it
        else:
            dummy = lambd[-1]
            lambd = lambd[:-1]
            return lambd, dummy, it
    
    def compute_stationary_distribution(self, M, init_pi=None, max_iters=10000, eps=1e-6):        
        pi = init_pi
        for it in range(max_iters):
            pi_ = pi @ M
            if np.linalg.norm(pi_ - pi) < eps:
                break
            pi = pi_
        return pi, it
    
    def compute_stationary_distribution_linalg(self, M):
        eps = 1.0 / np.max(np.abs((M)))
        mat = np.eye(M.shape[0]) + eps*M
        # Find the leading left eigenvector.
        vals, vecs = spsl.eigs(mat.T, k=1)
        res = np.real(vecs[:,0])
        res = (M.shape[0] / res.sum()) * res
        res = res/np.sum(res)
        return res[:, np.newaxis].T
    
    def compute_alpha(self, ranked_data):
        alpha = np.copy(self.a0)
        for data in ranked_data:
            for i in data[:-1]:
                alpha[i] += 1
        return alpha