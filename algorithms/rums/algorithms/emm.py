import numpy as np
import cvxpy as cp
import scipy as sp
from scipy.special import logsumexp
from rums.algorithms.pl import RegularizedILSR
from rums.algorithms.gmm import GMM
from scipy.linalg import svd
from sklearn.cluster import KMeans
from scipy.special import softmax
import rums
import abc


class EMM():
    """The Expectation-Majorization/Minorization method of Hunter
    """
    def __init__(self, n, K, mm_max_iters=10000):
        self.n = n
        self.K = K
        self.U_array = []
        self.alpha_array = []
        self.U = None
        self.F = lambda x: rums.algorithms.utils.F_gumbel(x, 1)
        self.F_prime = lambda x: rums.algorithms.utils.F_prime_gumbel(x, 1)
        self.mm_max_iters = mm_max_iters
        self.win_matrix = None

    def fit(self, rankings, U_init=None, max_iters=1000, eps=1e-4):
        if U_init is None:
            U_init, _ = self.random_init(rankings)

        assert(U_init.shape == (self.K, self.n))
        U = U_init
        alpha = np.ones((self.K,)) * 1./self.K
        self.U_array.append(U)
        self.alpha_array = [alpha]

        self.win_matrix = construct_win_matrix(self.n, rankings)

        for _ in range(max_iters):
            qz = self.e_step(rankings, U, alpha)
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

    def random_init(self, rankings):
        # This seems to make the convergence too random?
        alpha = np.ones((self.K,)) * 1./self.K
        U_all = np.random.normal(size=(self.K, self.n))
        U_all = U_all - np.mean(U_all, 1)[:, np.newaxis]
        return U_all, alpha
    
    def spectral_init(self, rankings):
        X = self.embed(rankings)
        clusters, _ = self.spectral_clustering(X)
        U_all = []
        for k in range(self.K):
            rankings_k = [ranking for idpi, ranking in enumerate(rankings) if clusters[idpi] == k]
            gmm = GMM(self.n, self.F, self.F_prime, step_size=0.1)
            Uk = gmm.fit(rankings_k)
            U_all.append(Uk)
        return np.array(U_all)

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
            Uk = mm_rankings(self.n, rankings, self.win_matrix, data_weights=posterior_dist[:, k], initial_params=current_U[k,  :], max_iters=self.mm_max_iters)
            U_all.append(Uk)
        return np.array(U_all)

    def e_step(self, rankings, U_all, alpha):
        m = len(rankings)
        K = len(U_all)
        qz = np.zeros((m, K))
        for k in range(K):
            qz[:, k] = self.estimate_log_likelihood_pl(rankings, U_all[k]) + np.log(alpha[k])
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


#######################################################################################
#           MAJORIZATION - MINORIZATION ALGORITHM FROM CHOIX
#######################################################################################


class MM():
    def __init__(self, n):
        self.n = n

    def fit(self, rankings, max_iters=10000, tol=1e-5):
        return mm_rankings(self.n, rankings, max_iters=max_iters, tol=tol)


class ConvergenceTest(metaclass=abc.ABCMeta):

    """Abstract base class for convergence tests.
    Convergence tests should implement a single function, `__call__`, which
    takes a parameter vector and returns a boolean indicating whether or not
    the convergence criterion is met.
    """

    @abc.abstractmethod
    def __call__(self, params, update=True):
        """Test whether convergence criterion is met.
        The parameter `update` controls whether `params` should replace the
        previous parameters (i.e., modify the state of the object).
        """


class NormOfDifferenceTest(ConvergenceTest):

    """Convergence test based on the norm of the difference vector.
    This convergence test computes the difference between two successive
    parameter vectors, and declares convergence when the norm of this
    difference vector (normalized by the number of items) is below `tol`.
    """

    def __init__(self, tol=1e-8, order=1):
        self._tol = tol
        self._ord = order
        self._prev_params = None

    def __call__(self, params, update=True):
        params = np.asarray(params) - np.mean(params)
        if self._prev_params is None:
            if update:
                self._prev_params = params
            return False
        dist = np.linalg.norm(self._prev_params - params, ord=self._ord)
        if update:
            self._prev_params = params
        return dist <= self._tol * len(params)


def log_transform(weights):
    """Transform weights into centered log-scale parameters."""
    params = np.log(weights)
    return params - params.mean()


def exp_transform(params):
    """Transform parameters into exp-scale weights."""
    weights = np.exp(np.asarray(params) - np.mean(params))
    return (len(weights) / weights.sum()) * weights


def _mm_rankings_acc(n_items, rankings, win_matrix, data_weights, params):
    """Inner loop of MM algorithm for ranking data.
    
    win matrix should be in {0,1}^{n*m} where win_matrix[i, l] = 1 if item i 'wins' in ranking l (not last)
    choice_tensor should be in {0,1}^{n, m, n} where choice_tensor[i, l, j] = 1 if item i wins over item j in ranking l 
    by default choice_tensor[i, l, i] = 1
    """
    weights = exp_transform(params)
    
    wins = win_matrix @ data_weights
    denoms = np.zeros(n_items, dtype=float)

    for idx, ranking in enumerate(rankings):
        # This should be of size
        cum_weights_sum = np.cumsum(weights.take(ranking)[::-1])[::-1]
        cum_weights_sum = np.cumsum(1./cum_weights_sum) # cum_weights_sum[i] = sum_{0 to i} 1./sum(weights for items below j)
        cum_weights_sum[-1] = cum_weights_sum[-2]
        denoms[ranking] += data_weights[idx] * cum_weights_sum

    return wins, denoms

def _mm(n_items, data, win_matrix, data_weights, initial_params, alpha, max_iter, tol, mm_fun):
    """
    Iteratively refine MM estimates until convergence.
    Raises
    ------
    RuntimeError
        If the algorithm does not converge after `max_iter` iterations.
    """
    if initial_params is None:
        params = np.zeros(n_items)
    else:
        params = initial_params
    converged = NormOfDifferenceTest(tol=tol, order=1)
    for _ in range(max_iter):
        nums, denoms = mm_fun(n_items, data, win_matrix, data_weights, params)
        params = log_transform((nums + alpha) / (denoms + alpha))
        if converged(params):
            return params
    return params


def construct_win_matrix(n_items, rankings):
    # Win matrix should be in {0,1}^{n * m}
    # where W[i, l] denotes whethere item i is NOT the last item
    m = len(rankings)
    W = np.zeros((n_items, m))
    for l, ranking in enumerate(rankings):
        for i in range(len(ranking)-1):
            W[ranking[i], l] = 1
    return W


def mm_rankings(n_items, data, win_matrix=None, data_weights=None, initial_params=None, alpha=0.0,
        max_iters=10000, tol=1e-5):
    """Compute the ML estimate of model parameters using the MM algorithm.
    This function computes the maximum-likelihood (ML) estimate of model
    parameters given ranking data (see :ref:`data-rankings`), using the
    minorization-maximization (MM) algorithm [Hun04]_, [CD12]_.
    If ``alpha > 0``, the function returns the maximum a-posteriori (MAP)
    estimate under a (peaked) Dirichlet prior. See :ref:`regularization` for
    details.
    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Ranking data.
    initial_params : array_like, optional
        Parameters used to initialize the iterative procedure.
    alpha : float, optional
        Regularization parameter.
    max_iter : int, optional
        Maximum number of iterations allowed.
    tol : float, optional
        Maximum L1-norm of the difference between successive iterates to
        declare convergence.
    Returns
    -------
    params : numpy.ndarray
        The ML estimate of model parameters.
    """
    if data_weights is None:
        data_weights = np.ones((len(data),))

    if win_matrix is None:
        win_matrix = construct_win_matrix(n_items, data)

    return _mm(n_items, data, win_matrix, data_weights, initial_params, alpha, max_iters, tol,
            _mm_rankings_acc)