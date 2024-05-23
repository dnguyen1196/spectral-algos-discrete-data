import numpy as np
INVALID_RESPONSE = -99999

def exponential_bernoulli_sample(gamma):
    assert(gamma >= 0)
    if 0 <= gamma <= 1:
        K = 1
        while(1):
            A = 1 if np.random.rand() < gamma/K else 0
            if A == 0:
                break
            else:
                K += 1
        return 1 if K % 2 == 1 else 0
    else:
        for k in range(1, int(np.floor(gamma) + 1)):
            B = exponential_bernoulli_sample(1)
            if B == 0:
                return 0
        return exponential_bernoulli_sample(gamma - np.floor(gamma))

def discrete_gaussian_sample(sigma):
    t = int(np.floor(sigma) + 1)
    
    while(1):
        U = np.random.choice(t)
        D = exponential_bernoulli_sample(U/t)
        if D == 0:
            # Restart
            return discrete_gaussian_sample(sigma)
        
        V = 0
        while(1):
            A = exponential_bernoulli_sample(1)
            if A == 0:
                break
            V += 1

        B = 1 if np.random.rand() < 1./2 else 0
        if B == 1 and U == 0 and V == 0:
            return discrete_gaussian_sample(sigma)
        Z = (1 - 2*B) * (U + t * V)
        C = exponential_bernoulli_sample((np.abs(Z)-sigma**2/t)**2/(2*sigma**2))
        
        if C == 0:
            return discrete_gaussian_sample(sigma)
        return Z
    
def zeroProb(alpha):
    #  this is the chance of "staying put at 0"
    if np.random.rand() < (1.0 - alpha)/(1.0 + alpha):
        return 0
    else:
        return 1

def signProb():
    if np.random.rand() < 0.5:
        return -1
    else:
        return 1

def twoSidedGeoDist(alpha):
    #  (1) Did we "leave 0"? [Y=1|N=0]          (3) +/-
    return zeroProb(alpha) * np.random.geometric(1-alpha) * signProb()


def construct_markov_chain_private(performances, mechanism="Gauss", lambd=1., epsilon=0.1, subsample_graph=None):
    m = len(performances)
    
    if subsample_graph is None:
        subsample_graph = np.ones((m, m))
    
    D = np.ma.masked_where(performances == INVALID_RESPONSE, performances)
    D_compl = 1. - D
    M = np.ma.dot(D, D_compl.T)
    
    A = np.ma.masked_where(performances == INVALID_RESPONSE, np.ones_like(performances))
    B = np.ma.dot(A, A.T)
    
    np.fill_diagonal(M, 0)
    np.nan_to_num(M, False)
    M = np.round(M)
    
    M_non_priv = np.copy(M)
    M_add = np.copy(M)
    
    # Add discrete Gaussian noise
    for i in range(m):
        for j in range(m):
            if j != i and M[i, j] != 0 and np.abs(subsample_graph[i, j] - 1) < 1e-6:
                if mechanism == "Gauss":
                    noise = discrete_gaussian_sample((1./epsilon))
                else:
                    noise = twoSidedGeoDist(np.exp(-epsilon))
                    
                M[i, j] = max(1, M[i, j] + noise)    
                M_add[i, j] = noise
                
            if j != i and np.abs(subsample_graph[i, j] - 0) < 1e-6:
                M_non_priv[i, j] = 0
                M[i, j] = 0
                M_add[i, j] = 0
                    
    # Add regularization to the 'missing' entries
    M = np.where(np.logical_or((M != 0), (M.T != 0)), M+lambd, M)
    
    # d = np.ma.sum(M, 1) + 1
    d = np.ones((m,)) * np.ma.max(np.ma.sum(M, 1) + 1)

    for i in range(m):
        di = d[i]
        M[i, :] /= di
        M[i, i] = 1. - np.sum(M[i, :])

    return M_non_priv, M_add, M, d


def spectral_estimate_private(A, mechanism="Gauss", lambd=1., epsilon=0.1, subsample_graph=None, max_iters=1000, eps=1e-5):
    _, _, M, d = construct_markov_chain_private(A, mechanism=mechanism, lambd=lambd, epsilon=epsilon, subsample_graph=subsample_graph)
    assert(not np.any(np.isnan(M)))
    from scipy.sparse import csc_matrix

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
        
    pi = pi.T
    # pi = np.maximum(pi, 1e-12)
    pi /= np.sum(pi)
    pi = (pi/d)/np.sum(pi/d)
    assert(not np.any(np.isnan(pi)))
    beta = np.log(pi)
    beta = beta - np.mean(beta)
    assert(not np.any(np.isnan(beta)))
    return beta

def subsampl_graph(m, p):
    subsample_graph = np.zeros((m, m))
    
    for i in range(m-1):
        for j in range(i+1, m):
            if np.random.rand() < p:
                subsample_graph[i, j] = 1
                subsample_graph[j, i] = 1
    
    return subsample_graph


def randomized_response(data, epsilon):
    # With probability e^epsilon/(1+e^epsilon), flip the label
    # For each user, we need to compute the number of responses, then divide epsilon by that number
    num_responses = np.sum(data != INVALID_RESPONSE, 0)    
    m, n = data.shape
    privatized_data = np.copy(data)
    
    for l in range(n):
        effective_epsilon = epsilon / num_responses[l] # The epsilon we should be using for each user to flip their responses
        rand_p = np.random.rand(m) # Flip the binary response with probability 1/(1+e^epsilon)
        privatized_data[:, l] = np.where(np.logical_and(rand_p < 1/(1+np.exp(effective_epsilon)), data[:, l] != INVALID_RESPONSE), 1 - data[:, l], data[:, l])
    
    return privatized_data


def find_effective_epsilon0_zgauss(overall_epsilon, overall_delta):
    # xi = overall_epsilon
    # epsilon_0 = np.sqrt(2 * xi)
    # return epsilon_0
    return overall_epsilon/np.sqrt(2 * np.log(1./overall_delta))


def find_effective_epsilon0_rr(overall_epsilon, overall_delta, n):
    # epsilon0 is the maximum epsilon0 such that
    # e^eps = 1 + (e^eps0 - 1)/(e^eps0+1) (8 sqrt(e^eps0 log (4/delta))/sqrt(n) + 8e^eps/n) # Need to solve for eps0
    # it also needs to satisfy eps0 <= log(n/16log(2/delta))
    def eps_estimate(eps0):
        eeps0 = np.exp(eps0)
        temp1 = (8 * np.sqrt(eeps0 * np.log(4/overall_delta)))/(np.sqrt(n))
        temp2 =  8 * eeps0/n
        return np.log(1 + (eeps0-1)/(eeps0+1) * (temp1 + temp2))
    
    # Just do bi-section search\
    lower = 0.00001
    upper = np.log(n/(16*np.log(2/overall_delta)))
    
    if upper < 0:
        return overall_epsilon
    
    while eps_estimate(lower) > overall_epsilon:
        lower /= 2
    
    while upper - lower > 1e-8:
        mid = (upper + lower)/2
        if eps_estimate(mid) < overall_epsilon:
            lower = mid
        else:
            upper = mid
    
    return mid


def randomized_response(data, epsilon):
    # With probability e^epsilon/(1+e^epsilon), flip the label
    # For each user, we need to compute the number of responses, then divide epsilon by that number
    num_responses = np.sum(data != INVALID_RESPONSE, 0)    
    m, n = data.shape
    privatized_data = np.copy(data)
    
    for l in range(n):
        effective_epsilon = epsilon / num_responses[l] # The epsilon we should be using for each user to flip their responses
        rand_p = np.random.rand(m) # Flip the binary response with probability 1/(1+e^epsilon)
        privatized_data[:, l] = np.where(np.logical_and(rand_p < 1/(1+np.exp(effective_epsilon)), data[:, l] != INVALID_RESPONSE), 1 - data[:, l], data[:, l])
    
    return privatized_data