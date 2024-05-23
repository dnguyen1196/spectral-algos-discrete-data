import numpy as np
import cvxpy as cp
import scipy as sp
import rums
from scipy.special import logsumexp
from rums.algorithms.pl import RegularizedILSR, RankCentrality
from scipy.linalg import svd
from sklearn.cluster import KMeans
from scipy.special import softmax
import time
from scipy.optimize import LinearConstraint, minimize
import faiss



class SpectralEM():
    def __init__(self, n, K, lambd=1., nu=1., ilsr_max_iters=1, gmm_lambd=0.01, init_method="lrmc", init_param={}, extra_refinement=False, trimmed_llh=False, trimmed_threshold=None, hard_e_step=False,):
        self.n = n
        self.K = K
        self.lambd = lambd
        self.U_array = []
        self.alpha_array = []
        self.delta_array = []
        self.U = None
        self.nu = nu
        self.ilsr_max_iters = ilsr_max_iters
        self.choice_tensor = None
        self.gmm_lambd = gmm_lambd
        assert (init_method in ["rc", "gmm", "cluster", "lrmc"])
        self.init_method = init_method
        self.init_param = init_param
        self.extra_refinement = extra_refinement
        self.trimmed_llh = trimmed_llh
        self.trimmed_threshold = 1./(n**2) if trimmed_threshold is None else trimmed_threshold
        self.hard_e_step = hard_e_step
        
    
    def settings(self):
        return {
            "lambd" : self.lambd,
            "nu" : self.nu,
            "ilsr_max_iters": self.ilsr_max_iters,
            "gmm_lambd" : self.gmm_lambd,
            "init_method": self.init_method,
            "extra_refinement": self.extra_refinement,
            "trimmed_llh" : self.trimmed_llh,
            "trimmed_threshold" : self.trimmed_threshold,
            "hard_e_step": self.hard_e_step,
        }

    def fit(self, rankings, U_init=None, max_iters=100, eps=1e-4, verbose=False):
        if U_init is None:
            if verbose:
                print("U_init not given, running spectral initialization ... ")
            start = time.time()
            U_init = self.spectral_init(rankings, verbose)
            if verbose:
                print("Spectral init took", time.time() - start)

        # Construct the choice tensor so we don't have to repeat this computation
        if self.choice_tensor is None:
            self.choice_tensor = RegularizedILSR(self.n, self.lambd, self.nu).construct_choice_tensor(rankings)

        assert(U_init.shape == (self.K, self.n))
        U = U_init
        alpha = np.ones((self.K,)) * 1./self.K
        self.U_array.append(U)
        self.alpha_array.append(alpha)
        self.delta_array = [np.inf]

        start = time.time()
        if verbose:
            print("Starting EM from initial solution ... ")
        U, alpha = self.em(rankings, U, alpha, eps, max_iters, verbose)
        # If doing extra refinement``
        if self.extra_refinement:
            self.ilsr_max_iters = 100 # Run more iterations
            U, alpha = self.em(rankings, U, alpha, eps, max_iters, verbose)
        if verbose:
            print(f"EM took {time.time() - start} seconds to converge, after {len(self.U_array)} iterations")

        self.U = np.copy(U)
        self.alpha = np.copy(alpha)
        return U, alpha
    
    
    def em(self, rankings, U, alpha, eps=1e-5, max_iters=1000, verbose=False):
        for it in range(max_iters):
            start = time.time()
            qz = self.e_step(rankings, U, alpha)
            if verbose:
                print(f"EM iter {it}, E-step took {time.time() - start} seconds")

            start = time.time()
            U_ = self.m_step(rankings, qz, U)
            if verbose:
                print(f"EM iter {it}, M-step took {time.time() - start} seconds")

            self.U_array.append(U)
            alpha = np.mean(qz, 0)
            alpha /= np.sum(alpha) # Make sure that it's normalized
            self.alpha_array.append(alpha)

            if verbose:
                print(f"EM iter {it}, ||U - U_prev||^2_F = {np.sum((U - U_)**2)}")

            delta = np.sum((U - U_)**2)
            self.delta_array.append(delta)

            if delta < eps:
                U = U_
                break
            U = U_

        return U, alpha
    
    def spectral_init(self, rankings, verbose=False):
        start = time.time()
        X = self.embed(rankings)
        if verbose:
            print(f"Spectral Init: Embedding took {time.time() - start} seconds")
        start = time.time()
        clusters, mus = self.spectral_clustering(X)
        if verbose:
            print(f"Spectral Init: Spectral Clustering took {time.time() - start} seconds")

        U_all = []
        start = time.time()
        for k in range(self.K):
            if self.init_method == "cluster":
                rankings_k = [ranking for idpi, ranking in enumerate(rankings) if clusters[idpi] == k]
                lsr = RegularizedILSR(self.n, self.lambd, self.nu)
                Uk = lsr.fit(rankings_k, max_iters=self.ilsr_max_iters)
            elif self.init_method == "gmm":
                # Use GMM to estimate the initial starting value, or why don't we use Rank Centrality to estimate initial?
                Uk = self.gmm_estimate(mus[k, :])
            elif self.init_method == "lrmc":
                Uk = self.lrmc_estimate(mus[k, :])
            else:
                # Or use RC to estimate the initial starting value
                Uk = self.rc_estimate(mus[k, :])
            
            U_all.append(Uk)
        if verbose:
            print(f"Learning after clustering took {time.time() - start} seconds")
        return np.array(U_all)
    
    def fill_in_center(self, mu):
        # Estimate the center
        P = np.eye(self.n)
        P[np.triu_indices(self.n, 1)] = mu        
        P = -P.T + P
        P += 1./2
        return P
    
    def rc_estimate(self, mu):
        P = self.fill_in_center(mu)
        rc = RankCentrality(self.n)
        return rc.fit_from_pairwise_probabilities(P, **self.init_param)
    
    def gmm_estimate(self, mu):
        P = self.fill_in_center(mu)
        np.fill_diagonal(P, 0)
        
        F = lambda x: rums.algorithms.utils.F_gumbel(x, 1)
        F_prime = lambda x: rums.algorithms.utils.F_prime_gumbel(x, 1)
        
        def gmm_loss(U):
            """
            The goal is to minimize 
            L = sum_{a != b} (P[a,b] F(Ua - Ub) - P[b,a] F(Ub - Ua))**2
            """
            Delta = np.outer(U, np.ones((self.n,))) - np.outer(np.ones((self.n,)), U) # Delta[a,b] = Ua - Ub (note how this is the opposite of the Delta in CML)
            f_delta = F_prime(Delta)
            F_delta = F(Delta)
            L = 1./2 * np.sum((P * F_delta - P.T * F_delta.T)**2) + 1./2 * np.sum(U**2) * self.gmm_lambd
            grad = np.zeros((self.n,))
            for i in range(self.n):
                grad[i] = 2 * np.sum((P[i, :] * F_delta[i, :] - P[:, i] * F_delta[:, i]) * (P[i, :] * f_delta[i, :] + P[:, i] * f_delta[:, i])) + self.gmm_lambd * U[i]
            return L, grad

        constraint = LinearConstraint(np.ones((1, self.n)), 0, 0)
        res = minimize(gmm_loss, np.zeros((self.n,)), constraints=[constraint], jac=True)
        U = res["x"]
        return U - np.mean(U)

    def lrmc_estimate(self, mu):
        P = self.fill_in_center(mu)
        np.fill_diagonal(P, 0)
        
        mask = np.where(np.logical_or(P < 0.001, P > 0.999), 0, 1) # Only include pairs that are not extreme valued
        M = np.log(P/(1.- P))
        U = cp.Variable((self.n, 1))
        reg_term = 0.5 * cp.square(cp.norm(U, 2))

        # Is there an issue with indeterminate hessian here?
        loss = cp.sum_squares(
                    cp.multiply(
                        mask,
                        M - \
                        (
                            - U @ np.ones((1, self.n)) \
                            + np.ones((self.n, 1)) @ U.T
                        )
                    )) + reg_term

        objective = cp.Minimize(loss)
        prob = cp.Problem(objective)
        prob.solve()
        U = U.value.flatten()
        U = U - np.mean(U)
        return U                

    def embed(self, rankings):
        # Embedd the rankings into {-1, +1} vectorization
        d = int(self.n * (self.n-1)/2)
        X = np.zeros((len(rankings), d), dtype=np.float32)
        
        S = np.zeros((self.n, self.n, len(rankings)))
        for idpi in range(len(rankings)):
            ranking = rankings[idpi]
            Xi = np.zeros((self.n, self.n))
            for idi in range(len(ranking)-1):
                for idj in range(idi+1, len(ranking)):
                    Xi[ranking[idj], ranking[idi]] = 1./2
                    S[ranking[idi], ranking[idj], idpi] = 1
            X[idpi, :] = Xi[np.triu_indices(self.n, 1)].flatten()
        self.choice_tensor = S
        return X

    def spectral_clustering(self, X):
        start = time.time()
        U, s, Vh = svd(X, full_matrices=False)
        s[self.K:] = 0
        Sigma = np.diag(s)
        Y = (U @ Sigma) @ Vh
        clustering = KMeans(self.K)
        clustering.fit(Y)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        return labels, centers

    def m_step(self, rankings, posterior_dist, U_current=None):
        if U_current is None:
            U_current = np.zeros((self.K, self.n))

        U_all = []
        for k in range(self.K):
            lsr = RegularizedILSR(self.n, self.lambd, self.nu)
            # Use the precomputed choice tensor to avoid wasteful computation
            Uk = lsr.fit(self.choice_tensor, sample_weights=posterior_dist[:, k], theta_init=U_current[k,:], max_iters=self.ilsr_max_iters, is_choice_tensor=True)
            U_all.append(Uk)
        return np.array(U_all)

    def e_step(self, rankings, U_all, alpha):
        m = len(rankings)
        K = len(U_all)
        qz = np.zeros((m, K))
        for k in range(K):
            qz[:, k] = self.estimate_log_likelihood_pl(rankings, U_all[k]) + np.log(alpha[k])
        qz = softmax(qz, 1)
        # Zero out very small posterior term
        if self.trimmed_llh:
            qz = np.where(qz < self.trimmed_threshold, 0, qz)
            qz /= qz.sum(1)[:, np.newaxis] # Re-normalize
        
        if self.hard_e_step:
            qz_hard = np.zeros((m, K))
            for l in range(m):
                qz_hard[np.random.choice(K, size=None, p=qz[l, :])] = 1
            qz = qz_hard
        
        return qz
    
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

