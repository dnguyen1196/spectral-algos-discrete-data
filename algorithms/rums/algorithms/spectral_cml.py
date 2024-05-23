import numpy as np
from scipy.stats import norm
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from scipy import sparse
from rums.algorithms.glrm.reg import QuadraticReg, LassoReg
from rums.algorithms.glrm.loss import HingeLoss, QuadraticLoss, HuberLoss
from rums.algorithms.glrm.convergence import Convergence
from rums.algorithms.glrm.glrm import GLRM
from scipy.sparse.linalg import svds
import cvxpy as cp
from rums.algorithms.matrix_completion import LRMC, RUM_LRMC, SVT


class SpectralCML():
    def __init__(self, n, phi=None, F=None, F_prime=None, rank_breaking="random", regularization="ell1",
                    lambd=0.001, solver="SLSQP", step_size=1, mc_method="rum_lrmc", nu=1):
        self.n = n
        if phi is None:
            self.phi = lambda d: norm.ppf(d, 0, np.sqrt(2))
        else:
            self.phi = phi
            
        if F is None:
            self.F = lambda x: norm.cdf(x, 0, np.sqrt(2))
        else:
            self.F = F
            
        if F_prime is None:
            self.F_prime = lambda x: norm.pdf(x, 0, np.sqrt(2))
        else:
            self.F_prime = F_prime

        self.rank_breaking_scheme = rank_breaking
        assert(rank_breaking in ["random", "full"])
        self.lambd = lambd
        self.solver = solver
        self.step_size = step_size
        assert(regularization in ["ell1", "ell2"])
        self.regularization = regularization
        assert(mc_method in ["svt", "lrmc", "rum_lrmc"])
        self.mc_method = mc_method
        self.nu = nu

    def fit(self, rankings):
        comparisons = self.rank_breaking(rankings, self.rank_breaking_scheme)
        P = self.construct_pairwise_matrix(comparisons, self.nu)
        U_init = self.spectral_init(P, self.mc_method)
        self.U = self.composite_likelihood_estimate(P, U_init, self.regularization, lambd=self.lambd)
        return self.U

    def rank_breaking(self, rankings, scheme="random"):
        comparisons = []

        if scheme == "random":
            for rank in rankings:
                # Randomly form m/2 comparisons
                random_pairs = np.random.permutation(len(rank) + 1 if len(rank) % 2 == 1 else len(rank)).reshape(-1, 2)
                for i, j in random_pairs:
                    if i < len(rank) and j < len(rank):
                        # Add (winner, loser) pair
                        comparisons.append((rank[i], rank[j]) if i < j else (rank[j], rank[i]))

        else:
            # Full breaking (not theoretically justified)
            for rank in rankings:
                for i in range(len(rank)-1):
                    for j in range(i+1, len(rank)):
                        comparisons.append(
                            (rank[i], rank[j])
                        )

        return comparisons

    def construct_pairwise_matrix(self, comparisons, nu=1):
        P = np.eye(self.n) * 1./2

        for winner, loser in comparisons:
            P[loser, winner] += 1

        for i in range(self.n - 1):
            for j in range(i+1, self.n):
                num_comps = P[i, j] + P[j, i]
                if num_comps > 0:
                    if P[i, j] == 0 or P[j, i] == 0:
                        P[i, j] = (P[i, j]+nu)/(num_comps+nu*2)
                        P[j, i] = (P[j, i]+nu)/(num_comps+nu*2)
                    else:
                        P[i, j] = P[i, j]/num_comps
                        P[j, i] = P[j, i]/num_comps                       

        return P

    def spectral_init(self, P, mc_method="rum_lrmc"):
        # Remove extreme values to avoid numerical issues
        if mc_method == "svt":
            svt = SVT(self.n, self.phi)
            return svt.decompose(P)
        elif mc_method == "lrmc":
            lrmc = LRMC(self.n, self.phi)
            return lrmc.decompose(P)
        else:
            lrmc = RUM_LRMC(self.n, self.phi)
            return lrmc.decompose(P)

    def composite_likelihood_estimate(self, P, U_init, regularization="ell1", lambd=0.001):
        # Estimate the parameter via CML
        np.fill_diagonal(P, 0)
        n = len(U_init)

        def sign_modified(x):
            return np.sign(x)

        if regularization == "ell2":
            regl = lambda x: np.sum(x**2)
            regl_grad = lambda x: x

        else:
            regl = lambda x: np.sum(np.abs(x))
            regl_grad = lambda x: sign_modified(x)

        def neg_loglik_func(U):
            Delta = np.outer(np.ones((self.n,)), U) - np.outer(U, np.ones((self.n,)))
            f_delta = self.F_prime(Delta)
            F_delta = self.F(Delta)
            lnF_delta = np.log(F_delta)
            neg_loglik = -np.sum(P * lnF_delta) + lambd * regl(U)
            partialL_partialU = P * f_delta/ F_delta

            grad = np.zeros((n,))
            for i in range(n):
                grad[i] = -np.sum(partialL_partialU[:, i]) + np.sum(partialL_partialU[i, :]) + lambd * regl_grad(U[i])
            return neg_loglik, self.step_size * grad
        
        constraint = LinearConstraint(np.ones((1, self.n)), 0, 0)
        res = minimize(neg_loglik_func, U_init, method=self.solver, constraints=[constraint], jac=True)
        U = res["x"]
        np.fill_diagonal(P, 0.5)
        return U - np.mean(U)
