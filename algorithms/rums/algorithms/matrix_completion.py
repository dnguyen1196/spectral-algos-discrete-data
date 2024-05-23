import numpy as np
import cvxpy as cp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from rums.algorithms.glrm.reg import QuadraticReg, LassoReg
from rums.algorithms.glrm.loss import HingeLoss, QuadraticLoss, HuberLoss
from rums.algorithms.glrm.convergence import Convergence
from rums.algorithms.glrm.glrm import GLRM

def create_mask(m, n, E, complement=False):
    if complement:
        mask = np.ones((m, n))
        for x, y in E:
            mask[x, y] = 0
    else:
        mask = np.zeros((m, n))
        for x, y in E:
            mask[x, y] = 1
    return mask


def rank_break(rankings, full_breaking=False):
    comparisons = []

    if not full_breaking:
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


def construct_pairwise_matrix(comparisons, n, nu=1):
    P = np.eye(n) * 1./2

    for winner, loser in comparisons:
        P[loser, winner] += 1

    for i in range(n - 1):
        for j in range(i+1, n):
            num_comps = P[i, j] + P[j, i]
            if num_comps > 0:
                # if P[i, j] == 0 or P[j, i] == 0:
                P[i, j] = (P[i, j]+nu)/(num_comps+nu*2)
                P[j, i] = (P[j, i]+nu)/(num_comps+nu*2)
                # else:
                #     P[i, j] = P[i, j]/num_comps
                #     P[j, i] = P[j, i]/num_comps            

    return P


class SVT():
    def __init__(self, n, phi, nu=1):
        self.n = n
        self.phi = phi
        self.nu = nu

    def fit(self, rankings, full_breaking=False):
        comparisons = rank_break(rankings, full_breaking)
        P = construct_pairwise_matrix(comparisons, self.n, self.nu)
        U = self.decompose(P)
        self.U = U
        return U

    def decompose(self, P):
        M = csr_matrix(P)
        M.data = self.phi(M.data)

        u, s, vh = svds(M, 2)
        Mhat = u @ np.diag(s) @ vh
        U = Mhat[0, :] - np.mean(Mhat[0, :])
        return U



class LRMC():
    def __init__(self, n, phi, lambd=0.01, nu=1):
        self.n = n
        self.phi = phi
        self.nu = nu
        self.lambd = lambd

    def fit(self, rankings, full_breaking=False):
        comparisons = rank_break(rankings, full_breaking)
        P = construct_pairwise_matrix(comparisons, self.n, self.nu)
        U = self.decompose(P, self.lambd)
        self.U = U
        return U
    
    def decompose(self, P, lambd=0.001, reg="ell2"):
        M = csr_matrix(P)
        M.data = self.phi(M.data)

        u, s, vh = svds(M, 2)
        X0 = u @ np.diag(np.sqrt(s)) 
        Y0 = np.diag(np.sqrt(s)) @ vh
        # append col of ones to X, row of zeros to Y
        mu = M.mean(0)
        X0 = np.hstack((X0, np.ones((X0.shape[0],1)))) # + C*randn(m,k+1) Unsure why we need to add random perturbation here could be heuristic - We can consider this
        Y0 = np.vstack((Y0, mu)) # + C*randn(k+1,n)

        loss = QuadraticLoss # Loss function

        r = 2 # Rank 2
        if reg == "ell2":
            regX, regY = QuadraticReg(lambd), QuadraticReg(lambd)
        else:
            regX, regY = LassoReg(lambd), LassoReg(lambd)

        c = Convergence(TOL = 1e-2, max_iters = 1000)
        missing_list = list(zip(*np.where(P == 0)))

        matrix_completion = GLRM(M, loss, regX, regY, r, [missing_list], c, X0=X0, Y0=[Y0])
        X, Y = matrix_completion.fit()

        X = np.array(X)
        Y = Y[0]
        M = X @ Y
        U = M[0, :]

        return U - np.mean(U)



class RUM_LRMC():
    def __init__(self, n, phi, lambd=0.001, nu=1):
        self.n = n
        self.lambd = lambd
        self.phi = phi
        self.nu = nu
        self.U = None


    def fit(self, rankings, full_breaking=False):
        comparisons = rank_break(rankings, full_breaking)
        P = construct_pairwise_matrix(comparisons, self.n, self.nu)
        U = self.decompose(P, self.lambd)
        self.U = U
        return U

    def decompose(self, P, lambd=0.001, reg="ell2"):
        M = csr_matrix(P)
        M.data = self.phi(M.data) # Applying mapping function

        # print(M.indices)
        E = list(zip(*M.nonzero()))
        m, n = M.shape
        mask = create_mask(m, n, E)
        np.fill_diagonal(mask, 0)

        M_dense = M.toarray()
        assert(not np.any(np.isnan(M_dense)))
        U = cp.Variable((self.n, 1))

        if reg == "ell2":
            reg_term = 0.5 * lambd * cp.square(cp.norm(U, 2))
        else:
            reg_term = lambd * cp.norm1(U)

        # Is there an issue with indeterminate hessian here?
        loss = cp.sum_squares(
                    cp.multiply(
                        mask,
                        M_dense - \
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



