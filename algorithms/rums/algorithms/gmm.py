from random import sample
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.stats import norm
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from scipy import sparse


class GMM():
    def __init__(self, n, F=None, F_prime=None, solver="SLSQP", step_size=1, lambd=0.001):
        self.n = n
        if F is None:
            self.F = lambda x: norm.cdf(x, 0, np.sqrt(2))
        else:
            self.F = F
            
        if F_prime is None:
            self.F_prime = lambda x: norm.pdf(x, 0, np.sqrt(2))
        else:
            self.F_prime = F_prime

        self.solver = solver
        self.step_size = step_size
        self.lambd = lambd

    def fit(self, rankings, sample_weights=None, U_init=None):
        if sample_weights is None:
            sample_weights = np.ones((len(rankings),))

        comparisons, comparison_weights = self.rank_breaking(rankings, sample_weights)
        P = self.construct_pairwise_matrix(comparisons, comparison_weights)
        if U_init is None:
            U_init = np.zeros((self.n,))
        self.U = self.gmm(P, U_init)
        return self.U

    def gmm(self, P, U_init):
        np.fill_diagonal(P, 0)

        def gmm_loss(U):
            """
            The goal is to minimize 
            L = sum_{a != b} (P[a,b] F(Ua - Ub) - P[b,a] f(theta_b - theta_a))**2
            """
            Delta = np.outer(U, np.ones((self.n,))) - np.outer(np.ones((self.n,)), U) # Delta[a,b] = Ua - Ub (note how this is the opposite of the Delta in CML)
            f_delta = self.F_prime(Delta)
            F_delta = self.F(Delta)
            L = 1./2 * np.sum((P * F_delta - P.T * F_delta.T)**2) + 1./2 * np.sum(U**2) * self.lambd
            grad = np.zeros((self.n,))
            for i in range(self.n):
                grad[i] = 2 * np.sum((P[i, :] * F_delta[i, :] - P[:, i] * F_delta[:, i]) * (P[i, :] * f_delta[i, :] + P[:, i] * f_delta[:, i])) + self.lambd * U[i]
            return L, self.step_size * grad

        constraint = LinearConstraint(np.ones((1, self.n)), 0, 0)
        res = minimize(gmm_loss, U_init, method=self.solver, constraints=[constraint], jac=True)
        U = res["x"]
        np.fill_diagonal(P,  0.5)
        return U - np.mean(U)

    def rank_breaking(self, rankings, sample_weights=None):
        if sample_weights is None:
            sample_weights = np.ones((len(rankings)),)

        comparisons = []
        comparisons_weights = []
        for idpi, rank in enumerate(rankings):
            for i in range(len(rank)-1):
                for j in range(i+1, len(rank)):
                    comparisons.append(
                        (rank[i], rank[j])
                    )
                    comparisons_weights.append(sample_weights[idpi])

        return comparisons, np.array(comparisons_weights)

    def construct_pairwise_matrix(self, comparisons, comparisons_weights=None):
        if comparisons_weights is None:
            comparisons_weights = np.ones((len(comparisons)),)
        P = np.eye(self.n) * 1./2

        for idx, (winner, loser) in enumerate(comparisons):
            P[loser, winner] += 1 * comparisons_weights[idx]

        for i in range(self.n - 1):
            for j in range(i+1, self.n):
                num_comps = P[i, j] + P[j, i]
                if num_comps > 0:
                    P[i, j] = P[i, j]/num_comps
                    P[j, i] = P[j, i]/num_comps

        return P