import numpy as np
from scipy.sparse import csc_matrix
from .ability_estimation import ability_mle

INVALID_RESPONSE = -99999


def construct_markov_chain(performances, lambd=1., return_B = False, same_d = False):
    m = len(performances)
    
    D = np.ma.masked_where(performances == INVALID_RESPONSE, performances)
    D_compl = 1. - D
    M = np.ma.dot(D, D_compl.T)
    
    A = np.ma.masked_where(performances == INVALID_RESPONSE, np.ones_like(performances))
    B = np.ma.dot(A, A.T)
    
    np.fill_diagonal(M, 0)
    np.nan_to_num(M, False)
    M = np.round(M)
    
    # Add regularization to the 'missing' entries
    M = np.where(np.logical_or((M != 0), (M.T != 0)), M+lambd, M)
    d = []
    
    if same_d:
        d = np.ones((m,)) * np.ma.max(np.ma.sum(B, 1))
    else:
        d = np.ma.sum(B, 1) + 1
    
    for i in range(m):
        # di = max(np.sum(B[i, :]), 1)
        # d.append(di)
        di = d[i]
        M[i, :] /= di
        M[i, i] = 1. - np.sum(M[i, :])

    if return_B:
        B = B.data
        np.fill_diagonal(B, 0)
        return M, d, B
    return M, d
    

def construct_markov_chain_accelerated(A, lambd=0.1):
    m, n = A.shape
    D = np.ma.masked_equal(A, INVALID_RESPONSE, copy=False)
    
    D_compl = 1. - D
    M = np.ma.dot(D, D_compl.T)
    np.fill_diagonal(M, 0)
    M = np.round(M)
    
    # Add regularization to the 'missing' entries
    M = np.where(np.logical_or((M != 0), (M.T != 0)), M+lambd, M)
    
    d = []
    for i in range(m):
        di = max(np.sum(M[i, :]), 1)
        d.append(di)
        M[i, :] /= max(d[i], 1)
        M[i, i] = 1. - np.sum(M[i, :])

    d = np.array(d)
    return M, d
    

def spectral_estimate(A, accelerated=True, max_iters=10000, return_beta=True, lambd=1, eps=1e-6):
    """Estimate the hidden parameters according to the Rasch model, either for the tests' difficulties
    or the students' abilities. Following the convention of Girth https://eribean.github.io/girth/docs/quickstart/quickstart/
    the rows correspond to the problems and the columns correspond to the students.
    """
    if accelerated:
        M, d = construct_markov_chain_accelerated(A, lambd=lambd)
    else:
        M, d = construct_markov_chain(A, lambd=lambd)

    M = csc_matrix(M)
    
    m = len(A)        
    pi = np.ones((m,)).T
    for _ in range(max_iters):
        pi_next = (pi @ M)
        pi_next /= np.sum(pi_next)
        if np.linalg.norm(pi_next - pi) < eps:
            pi = pi_next
            break
        pi = pi_next
        
    pi = pi.T/d
    if return_beta:
        beta = np.log(pi)
        beta = beta - np.mean(beta)
        return beta
    
    return pi/np.sum(pi)


def spectral_joint_estimate(A, max_iters=10000, lambd=1, eps=1e-6):
    m = len(A) # Number of items
    n = len(A[0]) # Number of users
    
    # Estimate one side of the parameters
    beta = spectral_estimate(A, True, max_iters, True, lambd, eps)
    
    # Then estimate the other side by solving n maximum likelihood estimate problem
    theta = ability_mle(A, beta, 1) # TODO: try different estimator, also try running the algorithm twice then learn a shift parameter
    return beta, theta