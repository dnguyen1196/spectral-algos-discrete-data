import pandas as pd
import numpy as np
from scipy.stats import norm as spnorm
from .poly_rasch import SpectralAlgorithm, estimate_abilities_given_difficulties
from scipy.sparse import csr_matrix, coo_matrix
import scipy as sp
import time 
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def evaluate_heldout_ratings_itemized(abilities, difficulties, heldout_scores):
    """Function to evaluate goodness of fit on some heldout dataset

    Args:
        abilities (vector of float of shape (n_users,)): Ability parameters
        difficulties (matrix of shape (n_items, n_levels)): Difficulty parameters
        heldout_scores (list of (user_id, item_id, score)): List of user-item scores

    Returns:
        MLE predicted scores, Actual given scores, log likelihood
    """
    from scipy.special import expit as logistic_sigmoid
    # Assume that heldout scores is a list of (user, movie, rating) tuples    
    # Compute likelihood as well
    K = difficulties.shape[1] # score goes from 0, 1, ..., K
    
    scores_pred = []
    scores_actual = []
    likelihood = []
    
    for (student,test,score) in heldout_scores:
        theta = abilities[student]
        thresholds = difficulties[test, :]

        ability_ms_diff = theta - thresholds

        # Pnk = logit(beta_n - theta_k) = prob(student n overcomes threshold k)
        p = logistic_sigmoid(ability_ms_diff)
        # Qnik = 1 - Pnik
        q = 1. - p # Complementary fail probability

        # Compute the running Pnik product
        # P(Xni = l) = passes all level from 1 to l, but fails from l+1 to L
        prob = np.zeros((K+1,))
        for l in range(K+1):
            # For the two extreme cases where l = 0, or L
            if l == 0:
                prob[l] = np.prod(q)
            elif l == K:
                prob[l] = np.prod(p)
            else:
                # p[:, l] = [Pn1, Pn2, ... , Pnl]
                # q[:, l:] = [Qnl+1,..., QnL]
                prob[l] = np.prod(p[:l]) * np.prod(q[l:])

        # Normalize by row to obtain grade probability
        prob /= np.sum(prob)
        assert(not np.any(prob == 0))

        # Predict MLE scores
        scores_pred.append(np.argmax(prob))
        scores_actual.append(score)
        likelihood.append(prob[int(score)])
        
    scores_pred = np.array(scores_pred)
    scores_actual = np.array(scores_actual)
    likelihood = np.array(likelihood)
    
    return scores_pred, scores_actual, likelihood


def evaluate_mixture(betas, thetas, q, ratings):
    """Evaluate a mixture model goodness of fit on some heldout ratings

    Args:
        betas (matrix of shape (n_items, n_levels)): Difficulty estimates
        thetas (vector of shape (n_users)): Ability estimates
        q (matrix of shape (n_users, n_components)): User-wise class membership probabilities
        ratings (list if (user, item, score)): Heldout ratings

    Returns:
        prediction error, log likelihood
    """
    C, m, K = betas.shape
    n, _ = thetas.shape
    assert(q.shape == (n, C))
    # For each validation ratings and each heldout ratings, compute both log-likelihood and mean absolute error
    n_ratings = len(ratings)
    
    likelihood = np.zeros((n_ratings, C))
    
    prediction_err = np.zeros((n_ratings, C))
    # To be consistent with single model, we do maximum likelihood estimate
    # Pick the most likely mixture component, then pick the most likely score
    posterior = np.array([q[l] for (l, _, _) in ratings])
    
    for c in range(C):
        thetas_c = thetas[:, c]
        betas_c = betas[c]
        scores_pred_c, scores_actual_c, likelihood_c = evaluate_heldout_ratings_itemized(thetas_c, betas_c, ratings)
        likelihood[:, c] = likelihood_c
        prediction_err[:, c] = np.abs(scores_pred_c - scores_actual_c)
        
    log_likelihood = np.sum(np.log(np.sum(likelihood * posterior, 1)))
    
    which_c = np.argmax(posterior, 1)
    pred_err = 0.
    for l, c in enumerate(which_c):
        pred_err += prediction_err[l, c]
    pred_err /= len(which_c)
    
    return pred_err, log_likelihood


def cartesian_product(*arrays):
    """ Compute set cartesian product of multiple sets
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def frobenius_diff(betas1, betas2):
    """ 
    Compute Frobenius squared norm of betas1 and betas2 modulo optimal row permutations
    """
    C, m, K = betas1.shape
    assert(betas2.shape == (C, m, K))
    
    min_err = None
    from itertools import permutations
    for pi in permutations(list(range(C))):
        err = 0.
        for i, pi_i in enumerate(pi):
            err += np.sum(np.square(betas1[i] - betas2[pi_i]))
            
        if min_err is None:
            min_err = err
        else:
            min_err = min(min_err, err)
                
    return min_err


class SpectralEM:
    def __init__(self, C, K, sigma=1., lambd=0.1):
        """_summary_

        Args:
            C (int): Number of mixture components
            K (int): Number of levels
            sigma (float, optional): User ability variance. Defaults to 1..
            lambd (float, optional): Regularization parameter for numerical stability. Defaults to 0.1.
        """
        self.sigma = sigma
        self.lambd = lambd
        self.W_tensor = {}
        self.sparse_matrices = {}
        self.C = C
        self.K = K
    
    
    def fit(self, X, 
            betas_init=None, 
            thetas_init=None, 
            sigma=None,
            max_iters=100, eps=1e-4, 
            validation_ratings=None, 
            auto_sigma=False,
            detailed_tracking=False):
        
        """
        
        Args:
            X: input response matrix
            betas_init: Optional initial difficulty guess, should be of shape (n_components, n_items, n_levels)
            thetas_init: Optional initial ability guess, should be of shape (n_users,)
            sigma: User ability variance
            max_iters, eps: max iterations and tolerance parameters
            validation_ratings: Optional heldout dataset for early stopping
            auto_sigma: Automatically adjust the user ability variance, this only works if betas_init is None
            detailed_tracking: if set to True, returns additional information traking the various quantities in all EM iterations            
            
        
        Returns:
            betas: Mixture of PCM estimates, a tensor of shape (n_components, n_items, n_levels)
            thetas: User parameter estimate
            q: Class membership probability estimate
            it: Number of iterations
        
        
        """        
        n, m = X.shape
        K = self.K
        C = self.C
        Z = self.construct_Z_tensor(X, K)
        sigma = self.sigma if sigma is None else sigma
        Z_masked = np.ma.masked_array(Z, mask=(Z==0))
        thetas = None
        
        q_track = []
        betas_track = []
        thetas_track = []
        val_llh = []

        if betas_init is None:
            betas, thetas, _, sigma_est = self.initialize(X, Z, Z_masked=Z_masked, C=C, K=K, sigma=sigma)
            if auto_sigma:
                sigma = sigma_est
                self.sigma = sigma_est
        else:
            betas = np.copy(betas_init)
                
        if thetas_init is None:
            thetas = np.zeros((n, C))
            for c in range(C):
                thetas[:, c] = self.estimate_thetas_mle(betas[c, :, :], X, Z)
        else:            
            thetas = np.copy(thetas_init)

        
        if validation_ratings is not None:
            q_init = self.E_step(X, Z_masked, betas, thetas, sigma=sigma)
            _, llh = evaluate_mixture(betas, thetas, q_init, validation_ratings)
            val_llh.append(llh)
        
        it = 0
        for it in range(max_iters):
            if C == 1:
                q = np.ones((n, C))
            else:
                q = self.E_step(X, Z_masked, betas, thetas, sigma, default_uniform=False)
            
            q_track.append(np.copy(q))
            betas_next, thetas = self.M_step(X, Z, Z_masked=Z_masked, q=q, K=K, sigma=sigma) 
            
            if validation_ratings is not None:
                _, llh = evaluate_mixture(betas_next, thetas, q, validation_ratings)
                if llh < val_llh[-1]: # Using validation likelihood as early stopping rule
                    break
                else:
                    val_llh.append(llh)
            
            betas_track.append(np.copy(betas_next))
            thetas_track.append(np.copy(thetas))
            if np.sum(np.square(betas_next - betas)) < eps:
                break
            betas = betas_next

        q = self.E_step(X, Z_masked, betas, thetas, sigma, default_uniform=True)
        if detailed_tracking:
            return betas, thetas, q, it, betas_track, thetas_track, q_track
        return betas, thetas, q, it
    
    
    def initialize(self, X, Z, Z_masked, C, K, sigma=1.0):
        """Initialize the betas estimate

        Args:
            X (matrix of shape (n_users, n_items)): Input response data
            Z (tensor of shape (n_users, n_items, n_items)): Special data structure for efficient computation
            Z_masked: masked array of Z for faster computation
            C (int): Number of components
            K (int): Number of levels
            sigma (float, optional): _description_. Defaults to 1.0.

        Returns:
            betas_init: initial betas estimate 
            thetas_init: initial thetas (ability) estimate
            labels: Class membership estimate
            sigma_est: estimate of user parameter std
        """
        n, m  = X.shape
        if C >= 2:
            pairwise_embeddings = []
            # If the number of items is too large, pick the top 50 items with the most number of ratings
            cutoff = 100
            if m > cutoff: # Heuristic: Focus on the subset of X with the highest variance in ratings
                score_variance = np.ma.array(X, mask=(X==-99999)).var(axis=0)
                idx = np.argsort(score_variance)[-cutoff:]
                X_subset = np.take(X, idx, axis=1)
            else:
                X_subset = X
            
            for l in range(n):
                # Pairwise difference between missing/non-missing response = 0
                pairwise_diff = X_subset[l].reshape((-1, 1)) - X_subset[l].reshape((1, -1))
                pairwise_diff = np.where(np.logical_or(pairwise_diff < -1000, pairwise_diff > 1000), 0, pairwise_diff)
                pairwise_embeddings.append(pairwise_diff.flatten())
            
            # Do a low rank approximation
            # Run K-means on embeddings
            kmeans = KMeans(C)
            kmeans.fit(pairwise_embeddings)
            labels = kmeans.labels_
        else:
            labels = [0 for _ in range(len(X))]
        
        q = np.zeros((n, C))
        for l, c in enumerate(labels):
            q[l,c] = 1.
            
        betas_init, thetas_init = self.M_step(X, Z, Z_masked=Z_masked, q=q, K=K, sigma=sigma)
        sigma_est = np.std([thetas_init[l,c] for l, c in enumerate(labels)])
        return betas_init, thetas_init, labels, sigma_est
    
    
    def E_step(self, X, Z_masked, betas, thetas, sigma=1.0, default_uniform=True):
        """
        Perform the E-step of the EM algorithm
        
        """
        C, m, K = betas.shape
        n, _ = X.shape
        q = np.zeros((n, C))
        
        for c in range(C):
            betas_c = betas[c]
            # Use the thetas together with betas to compute likelihood of responses
            thetas_c = thetas[:, c]
            response_prob_c = self.compute_response_likelihood(betas_c, thetas_c)
            # This will have shape (n, m, K+1)
            # likelihood_c should have shape (n,)
            likelihood_c = np.ma.prod(np.ma.prod(Z_masked * response_prob_c, axis=-1), axis=-1) * spnorm.pdf(thetas_c, 0, sigma)
            q[:, c] = likelihood_c
        
        # Then normalize to obtain posterior distribution
        # norm_sum = np.sum(q, axis=1)
        # # q = q / norm_sum.reshape((-1, 1))     
        norm_sum = np.sum(q, axis=1)
        q = q / norm_sum.reshape((-1, 1))
        if default_uniform:
            q = np.where(np.isnan(q), 1./C, q)
        else:
            q = np.where(np.isnan(q), 0., q)
        return q
    
    
    def M_step(self, X, Z, Z_masked, q, K, sigma=1.0):
        """
        Performs the M-step of the EM algorithm. It consists of two phases
        Phase 1 estimates the normalized betas for each component
        Phase 2 estimates the shift for each component        
        
        """
        n, C = q.shape
        # For each cluster, estimate normalized betas
        # Run Spectral algo on X with weight q
        # Estimate theta for each user
        # Estimate the shift for each component using weighted mean
        # betas_next has shape (C, m, K)
        betas_next = self.weighted_spectral_estimate(X, q, K)
        thetas_estimate = []
        
        # Estimate the shift
        for c in range(C):
            betas_c = betas_next[c]
            q_c = q[:, c]
            shift = -self.estimate_shift_mmle(X, Z_masked=Z_masked, betas_c=betas_c, q_c=q_c, sigma_c=sigma)
            thetas_c = self.estimate_thetas_mle(betas_c + shift, X, Z)
            thetas_estimate.append(thetas_c)
            betas_next[c] = betas_c + shift
        
        thetas_estimate = np.array(thetas_estimate).T # Will have shape (n, C)
        return betas_next, thetas_estimate
        
    
    def compute_response_likelihood(self, betas_c, thetas_c):
        """
        Estimate the response likelihood of a PCM with parameters betas_c and thetas_c
        
        ---
        Returns a (n, m, K+1) tensor where P[l, i, k] = probability that user l scores item i with k
        
        """
        n = len(thetas_c)
        m, K = betas_c.shape
        
        taus = np.cumsum(betas_c, 1)
        taus = np.hstack((np.zeros((m, 1)), taus)) # Augment a 0's
        
        # kTheta will have shape (n, K+1)
        # where kTheta[l,k] = k * theta_l
        kTheta = thetas_c.reshape((-1, 1)) @ np.arange(K+1).reshape((1, -1))
        Delta = kTheta.reshape((n, 1, K+1)) - taus # Should have shape (n, m, K+1) 
        expDelta = np.exp(Delta)
        # Normalize to obtain valid response probability
        probs = expDelta / np.sum(expDelta, -1).reshape((n, m, 1)) # Shape (n, m, K+1)
        return probs
    
    
    def estimate_shift_mmle(self, X, Z_masked, betas_c, q_c, sigma_c=1.0, grid_size=500):
        """
        Estimate the shift of the betas parameter.
        
        Z_masked has shape (n, m, K+1)
        Find delta such to maximize the marginal log likelihood of         
        log E_{theta ~ N(delta, sigma^2)} Pr(X | betas_c, theta)
        
        """
        n, _ = X.shape
        m, K = betas_c.shape
        taus = np.cumsum(betas_c, 1)
        taus = np.hstack((np.zeros((m, 1)), taus)) # Augment a 0's
        Z_weighted = Z_masked * q_c.reshape((n, 1, 1)) # Weighting by the posterior probability
        
        def approximate_marginal_llh(delta):
            thetas_sample = np.linspace(delta - 4 * sigma_c, delta + 4 * sigma_c, grid_size)
            weights = spnorm.pdf(thetas_sample, delta, sigma_c)
            weights /= np.sum(weights)

            kTheta = thetas_sample.reshape((-1, 1)) @ np.arange(K+1).reshape((1, -1))
            Delta = kTheta.reshape((grid_size, 1, K+1)) - taus # Should have shape (grid_size, m, K+1) 
            expDelta = np.exp(Delta)
                  
            probs = expDelta / np.sum(expDelta, -1).reshape((grid_size, m, 1)) # Shape (grid_size, m, K+1)
            expected_probs = probs * weights.reshape((grid_size, 1, 1))        # Shape (grid_size, m, K+1)
            expected_probs = expected_probs.sum(0)                             # Shape (m, K+1)
            
            likelihood = np.ma.prod(np.ma.prod(Z_weighted * expected_probs, axis=-1), axis=-1) # Multiplying (n, m, K+1) with (m, K+1) followed by multiplying over last two axis
            return -np.sum(np.log(likelihood))
        
        # Set up a scalar optimization problem using scipy
        res = minimize(approximate_marginal_llh, 0.1)
        return res.x[0]
        
    
    def weighted_spectral_estimate(self, X, q, K=None, max_mc_iters=1000, mc_eps=1e-4):
        """
        
        ---
        Returns: betas which have shape (C, m, K)
        
        """ 
        hat_betas = self.weighted_spectral_normalized_estimate(X, q, K, max_mc_iters=1000, mc_eps=1e-4)
        betas = self.weighted_spectral_unnormalized_estimate(X, q, hat_betas)
        return betas

    
    def weighted_spectral_unnormalized_estimate(self, X, q, hat_betas):
        """
        
        Obtain unnormalized estimate 
        
        """
        n, C = q.shape
        _, m, K = hat_betas.shape
        betas = np.copy(hat_betas)
        hat_betas_1 = hat_betas[:, :, 0]
        
        for k in range(2, K+1):            
            Y_1km1 = np.zeros((C, m, m))
            Y_0k = np.zeros((C, m, m))
            for c in range(C):
                q_c = q[:, c]
                Y_1km1[c] = self.construct_W_matrix_weighted(X, 1, k-1, q_c)
                Y_0k[c] = self.construct_W_matrix_weighted(X, 0, k, q_c)
            
            numer = Y_1km1 * np.exp(hat_betas_1).reshape((C, m, 1))
            # hat_betas_k has shape (C, m)
            hat_betas_k = hat_betas[:, :, k-1]
            denom = Y_0k * np.exp(hat_betas_k).reshape((C, 1, m))

            delta_k = np.log(numer.sum(-1).sum(-1)/denom.sum(-1).sum(-1)) # Should have shape (C,)
            delta_k = np.where(np.isinf(delta_k), 0., delta_k)
            betas[:, :, k-1] = hat_betas_k + delta_k.reshape((C, 1))    
            
        return betas
    
    
    def weighted_spectral_normalized_estimate(self, X, q, K=None, max_mc_iters=2000, mc_eps=1e-4):
        """
        
        Compute the normalized spectral estimate from the response data and the posterior probabilities
        
        Args:
            X: response data as a matrix of shape (n_users, n_item)
            q: Class posterior probabilities as a matrix of shape (n_users, n_components)
            K: Number of levels
        
        
        """
        if K is None:
            K = int(np.max(X))
        n, C = q.shape
        _, m = X.shape
        
        hat_betas = []
        
        # Estimate params for each successive level
        for k in range(1, K+1):
            Y_kk1 = np.zeros((C, m, m))
            for c in range(C):
                q_c = q[:, c]
                Y_kk1[c] = self.construct_W_matrix_weighted(X, k, k-1, q_c)
                        
            # Add regularization
            if self.lambd > 0:
                Y_kk1 = np.where(np.logical_or(Y_kk1 != 0, np.transpose(Y_kk1, [0, 2, 1]) != 0), Y_kk1 + self.lambd, Y_kk1)
            
            # row_sums will have shape (C, m)
            row_sums = np.maximum(Y_kk1.sum(-1), np.ones((C, m)))
            M = Y_kk1 / row_sums.reshape((C, m, 1))
            # M[b] has shape (m, m)
            for c in range(C):
                np.fill_diagonal(M[c], 1. - np.sum(M[c], 1))
                
            # Verify M                 
            # Compute stationary distribution
            pi = np.ones((C, m, 1)) / m
            # M_transpose still have shape (C, m, m)
            # But we transpose each slice of M
            M_transpose = np.transpose(M, [0, 2, 1])
            
            for _ in range(max_mc_iters):
                pi_next = M_transpose @ pi
                if np.max(np.abs(pi_next - pi)) < mc_eps:
                    break
                pi = pi_next
                pi = pi / pi.sum(-1).sum(-1).reshape((C, 1, 1))
                
            # Divide each pi[c, i] by row_sums[c,i]
            pi = pi / row_sums.reshape((C, m, 1))
            # Normalize
            pi = pi / pi.sum(-1).sum(-1).reshape((C, 1, 1))
                
            # Final pi will have shape (C, m, 1)
            pi = pi.reshape((C, m))
            log_pi = np.log(pi)
            
            # Will have shape (C, m) and row sum is 0
            normalized_betas = log_pi - np.mean(log_pi, axis=1).reshape((-1, 1))
            hat_betas.append(normalized_betas)
        
        # hat_betas currently has shape (K, C, m)
        # We swap first and second axis
        hat_betas = np.array(hat_betas)
        hat_betas = np.moveaxis(hat_betas, 0, 1) # Swap the first two axis to have shape (C, K, m)   
        return np.moveaxis(hat_betas, 1, 2)

    
    def construct_Z_tensor(self, X, K):
        """ 
        Construct a tensor Z of shape (n, m, K+1)
        where Z_{lik} = 1[X_{li} = k]
        """
        n, m = X.shape
        # Z will have shape (n, m, K+1)
        Z = np.zeros((n, m, K+1))
        for l in range(n):
            for i in range(m):
                if X[l][i] != -99999:
                    Z[l, i, int(X[l][i])] = 1.
        return Z
    
    
    def construct_W_matrix_weighted(self, X, k1, k2, q):
        """
        Save the sparse matrix?
        ---
        returns a single matrix of shape (m, m)
        """        
        if k1 not in self.sparse_matrices:
            self.sparse_matrices[k1] = csr_matrix((X == k1).astype(float))
        if k2 not in self.sparse_matrices:
            self.sparse_matrices[k2] = csr_matrix((X == k2).astype(float))
            
        is_k1 = self.sparse_matrices[k1].toarray() * q.reshape((-1, 1))
        is_k2 = self.sparse_matrices[k2].toarray() * q.reshape((-1, 1))
        Y = (is_k1.T @ is_k2)
        return Y

    
    def construct_W_flatten_matrix(self, X, k1, k2):
        """
        Return a matrix W of shape (n, m * m) which is the flattened version of a tensor of Z of shape [n, m, m] such that
        Z[l, i, j] = 1 if X_li = k1, X_lj = k2
        
        IndicK1 = (Z == k1).astype(int)
        IndicK2 = (Z == k2).astype(int)
        
        Z = IndicK1.reshape((n, m, 1)) * IndicK2.reshape((n, 1, m))
        
        """
        n, m = X.shape
        W = sp.sparse.csr_array((n, m*m))
        for l in range(n):
            Sl_k1 = np.argwhere(X[l] == k1).flatten()
            Sl_k2 = np.argwhere(X[l] == k2).flatten()
            pos = cartesian_product(Sl_k1, Sl_k2)
            indices = pos[:, 0] * m + pos[:, 1]
            W[[l], indices] = 1.
        return sp.sparse.coo_array(W)

    
    def estimate_thetas_mle(self, betas_c, X, Z):
        return estimate_abilities_given_difficulties(X, betas_c)
        
