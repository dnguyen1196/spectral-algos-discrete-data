import numpy as np
import cvxpy as cp
import scipy as sp
from scipy.special import logsumexp
from rums.algorithms.pl import RegularizedILSR
from rums.algorithms.gmm import GMM
from scipy.optimize import LinearConstraint
from scipy.linalg import svd
from sklearn.cluster import KMeans
from scipy.special import softmax
import rums
from scipy.optimize import minimize



class GMM_kPL():
    def __init__(self, n, K, lambd=0.):
        self.n = n
        self.K = K
        self.lambd = lambd
        self.F = lambda x: rums.algorithms.utils.F_gumbel(x, 1)
        self.F_prime = lambda x: rums.algorithms.utils.F_prime_gumbel(x, 1)

    def fit(self, rankings, U_init=None, alpha_init=None):
        if alpha_init is None:
            alpha_init = np.ones((self.K,)) * 1./self.K
        if U_init is None:
            U_init, _ = self.random_init(rankings)
        assert(U_init.shape == (self.K, self.n))
        return self.gmm_kpl(self.embed(rankings), U_init, alpha_init)
    
    def random_init(self, rankings):
        alpha = np.ones((self.K,)) * 1./self.K
        U_all = np.random.normal(size=(self.K, self.n))
        U_all = U_all - np.mean(U_all, 1)[:, np.newaxis]
        return U_all, alpha

    def embed(self, rankings):
        # Embedd the rankings into {-1, +1} vectorization
        d = self.n**2
        X = np.zeros((len(rankings), d))
        for idpi, ranking in enumerate(rankings):
            Xi = np.zeros((self.n, self.n))
            for idx, i in enumerate(ranking[:-1]):
                for idj in range(idx+1, len(ranking)):
                    j = ranking[idj]
                    Xi[j, i] = 1
                    Xi[i, j] = 0
            X[idpi, :] = Xi.flatten()
        return X

    def gmm_kpl(self, X, U, alpha):
        U_all = []

        # Compute the
        P = np.mean(X, 0)
        assert(P.shape == (self.n * self.n,))

        def moment_loss(U_alpha):
            loss = 0. 
            U = U_alpha[:-self.K]
            alpha = U_alpha[:-self.K]
            # Reshape U into
            U_all = U.reshape((self.K, self.n))
            P_estimate =  np.zeros((self.n**2,))

            for k in range(self.K):
                Uk = U_all[k, :]
                U_delta_k = np.outer(Uk, np.ones((self.n,))) - np.outer(np.ones((self.n,)), Uk) 
                # Note: U_delta_k[a,b] = Ua - Ub
                P_estimate_k = self.F(U_delta_k).flatten() * alpha[k]
                assert(P_estimate_k.shape == (self.n**2,))
                P_estimate += P_estimate_k
            
            loss += 1./2 * np.sum((P - P_estimate)**2) + 1./2 * np.sum(U**2) * self.lambd
            return loss

        # Add constraint to make sure alpha sums to 1
        mask = np.zeros((1, self.n * self.K + self.K))
        mask[0, -self.K:] = 1
        constraint = LinearConstraint(mask, 1, 1) # Make sure alpha sums to 1
        
        init_sol = np.zeros((self.n * self.K + self.K,))
        init_sol[:-self.K] = U.flatten()
        init_sol[-self.K:] = alpha

        res = minimize(moment_loss, init_sol, constraints=[constraint])
        U_alpha = res["x"]
        U_all = U_alpha[:-self.K].reshape((self.K, self.n))
        U_all -= np.mean(U_all, axis=1)[:, np.newaxis] # Normalize so that the utilities sum to 0
        alpha = U_alpha[-self.K:]
        alpha /= alpha.sum()
        return U_all, alpha


class GMM_kPL_Generalized(GMM_kPL):
    def __init__(self, n, K, lambd=0.):
        super(GMM_kPL_Generalized, self).__init__(n, K, lambd)

    def gmm_kpl(self, X, U, alpha):
        U_all = []

        # Compute the
        P = np.mean(X, 0)
        assert(P.shape == (self.n * self.n,))

        def moment_loss(U_alpha):
            loss = 0. 
            U = U_alpha[:-self.K]
            alpha = U_alpha[:-self.K]
            # Reshape U into
            U_all = U.reshape((self.K, self.n))
            P_estimate =  np.zeros((self.n**2,))

            for k in range(self.K):
                Uk = U_all[k, :]
                U_delta_k = np.outer(Uk, np.ones((self.n,))) - np.outer(np.ones((self.n,)), Uk) 
                # Note: U_delta_k[a,b] = Ua - Ub
                P_estimate_k = self.F(U_delta_k).flatten() * alpha[k]
                assert(P_estimate_k.shape == (self.n**2,))
                P_estimate += P_estimate_k
            
            loss += 1./2 * np.sum((P - P_estimate)**2) + 1./2 * np.sum(U**2) * self.lambd
            return loss

        # Add constraint to make sure alpha sums to 1
        mask = np.zeros((1, self.n * self.K + self.K))
        mask[0, -self.K:] = 1
        constraint = LinearConstraint(mask, 1, 1) # Make sure alpha sums to 1
        
        init_sol = np.zeros((self.n * self.K + self.K,))
        init_sol[:-self.K] = U.flatten()
        init_sol[-self.K:] = alpha

        res = minimize(moment_loss, init_sol, constraints=[constraint])
        U_alpha = res["x"]
        U_all = U_alpha[:-self.K].reshape((self.K, self.n))
        U_all -= np.mean(U_all, axis=1)[:, np.newaxis] # Normalize so that the utilities sum to 0
        alpha = U_alpha[-self.K:]
        alpha /= alpha.sum()
        return U_all, alpha


class EM_GMM_PL():
    def __init__(self, n, K, step_size=0.1, lambd=0.):
        self.n = n
        self.K = K
        self.U_array = []
        self.alpha_array = []
        self.U = None
        self.F = lambda x: rums.algorithms.utils.F_gumbel(x, 1)
        self.F_prime = lambda x: rums.algorithms.utils.F_prime_gumbel(x, 1)
        self.step_size = step_size
        self.lambd = lambd

    def fit(self, rankings, U_init=None, max_iters=100, eps=1e-4, random_init=True):
        # U, alpha = self.gmm_init(rankings)
        if U_init is None:
            U, alpha = self.spectral_init(rankings)
        else:
            U = U_init
            alpha = np.ones((self.K,)) * 1./self.K

        assert(U.shape == (self.K, self.n))
        self.U_array.append(U)
        self.alpha_array.append(alpha)

        for _ in range(max_iters):
            qz = self.e_step(rankings, U,  alpha)
            U_ = self.m_step(rankings, qz, U)

            self.U_array.append(U)
            alpha = np.mean(qz, 0)
            alpha /= np.sum(alpha)
            self.alpha_array.append(alpha)
            
            if np.sum((U - U_)**2) < eps:
                U = U_
                break
            U = U_

        self.U = np.copy(U)
        self.alpha = np.copy(alpha)
        return U, alpha
    
    def spectral_init(self, rankings):
        X = self.embed(rankings)
        clusters, yhat = self.spectral_clustering(X)
        U_all = []
        alpha = []
        for k in range(self.K):
            rankings_k = [ranking for idpi, ranking in enumerate(rankings) if clusters[idpi] == k]
            alpha.append(float(len(rankings)))
            gmm = GMM(self.n, self.F, self.F_prime, step_size=self.step_size, lambd=self.lambd)
            Uk = gmm.fit(rankings_k)
            U_all.append(Uk)

        alpha = np.array(alpha)
        alpha /= alpha.sum()
        return np.array(U_all), alpha

    def random_init(self, rankings):
        alpha = np.ones((self.K,)) * 1./self.K
        U_all = np.random.normal(size=(self.K, self.n))
        U_all = U_all - np.mean(U_all, 1)[:, np.newaxis]
        return U_all, alpha

    def gmm_init(self, rankings):
        gmm = GMM_kPL(self.n, self.K)
        return gmm.fit(rankings)

    def embed(self, rankings):
        # Embedd the rankings into {-1, +1} vectorization
        d = self.n**2
        X = np.zeros((len(rankings), d))

        for idpi, ranking in enumerate(rankings):
            Xi = np.zeros((self.n, self.n))
            for idx, i in enumerate(ranking[:-1]):
                for idj in range(idx+1, len(ranking)):
                    j = ranking[idj]
                    Xi[j, i] = 1
                    Xi[i, j] = -1
            X[idpi, :] = Xi.flatten()
        return X

    def spectral_clustering(self, X):
        U, s, Vh = svd(X, full_matrices=False)
        s[self.K:] = 0
        Sigma = np.diag(s)
        Y = (U @ Sigma) @ Vh
        assert(Y.shape == X.shape)
        clustering = KMeans(self.K)
        clustering.fit(Y)
        return clustering.labels_, clustering.cluster_centers_

    def m_step(self, rankings, posterior_dist, current_U):
        U_all = []
        for k in range(self.K):
            gmm = GMM(self.n, self.F, self.F_prime, step_size=self.step_size, lambd=self.lambd)
            Uk = gmm.fit(rankings, sample_weights=posterior_dist[:, k], U_init=current_U[k,  :])
            U_all.append(Uk)
        return np.array(U_all)

    def e_step(self, rankings, U_all, prior):
        m = len(rankings)
        K = len(U_all)
        qz = np.zeros((m, K))
        for k in range(K):
            qz[:, k] = self.estimate_log_likelihood_pl(rankings, U_all[k]) + np.log(prior[k])
        return softmax(qz, 1)
    
    def estimate_log_likelihood_pl(self, rankings, U):
        pi = softmax(U)
        log_likelihoods = []
        for ranking in rankings:
            pi_sigma = np.array([pi[i] for i in ranking])
            # Use cumsum here
            sum_pi = np.cumsum(pi_sigma[::-1])[::-1]
            log_lik = np.log(pi_sigma/sum_pi)
            log_lik = np.sum(log_lik[:-1])
            log_likelihoods.append(log_lik)
        return np.array(log_likelihoods)

