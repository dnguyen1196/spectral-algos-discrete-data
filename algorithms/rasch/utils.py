import numpy as np
from sklearn.metrics import roc_auc_score
from girth import ability_mle, ability_map, ability_eap
import time
from scipy.integrate import quadrature
from scipy.stats import norm
from scipy.special import expit


INVALID_RESPONSE = -99999


def graded_to_binary(A, f):
    for i in range(len(A)):
        for j in range(A.shape[1]):
            A[i, j] = f(A[i, j]) if A[i, j] != INVALID_RESPONSE else INVALID_RESPONSE
    
    return A


def partition_data(A, p_train=0.8, p_test=0.2, seed=None):
    n = A.shape[1]
    test_data = []
    np.random.seed(seed)
    
    perm = np.random.permutation(n)
    n_train = int(n*p_train)
    n_test = int(n*p_test)
    
    subset_cols_train = perm[:n_train]
    subset_cols_test = perm[-n_test:]
    A_train = A[:, subset_cols_train]
    
    # For each student, removes a portion of the data
    for j in subset_cols_test:
        responses = A[:, j]
        responses_idx = np.where(responses != INVALID_RESPONSE)[0]
        for i in responses_idx:
            test_data.append(((i, j), responses[i]))
    
    return A_train, test_data


def extract_binary_responses(A):
    binary_responses = []
    
    # For each student, removes a portion of the data
    for i in range(len(A)):
        responses_toi = A[i, :]
        responses_idx = np.where(responses_toi != INVALID_RESPONSE)[0]
        for j in responses_idx:
            # ((item, user), response)
            binary_responses.append(((i, j), responses_toi[j]))
    
    return binary_responses


def cramer_rao_lower_bound(beta_tests, theta_students, p):
    # The log likelihood should factor into individual tests
    m = len(beta_tests)
    n = len(theta_students)
    var = []
    for i in range(m):
        I_betai = 0.
        betai = beta_tests[i]

        for l in range(n):
            thetal = theta_students[l]
            I_betai += p * np.exp(-(thetal-betai))/(1+np.exp(-(thetal- betai)))**2
        var.append(1./I_betai)
    return np.array(var)

def quadrature_p_response(betas, sigma=1, mean=0, kappa=10):
    p_response = np.zeros((len(betas,)))
    for i, beta in enumerate(betas):
        func = lambda x: norm.pdf(x, mean, sigma) * expit(x-beta)
        p = quadrature(func, mean-kappa * sigma, mean+kappa * sigma)[0]
        p_response[i] = min(p, 1.)
    return p_response

def log_likelihood_heldout(p_response, test_data):
    # Compute the log likelihood on held out dataset
    preds = np.array([p_response[i] for (i, _), _ in test_data])
    y = np.array([response for (_, _), response in test_data])
    return np.mean(y * np.log(preds) + (1-y) * np.log(1-preds))

def bayesian_auc(p_response, test_data):
    # Compute AUC on heldout dataset
    y_score = [p_response[i]
        for (i, _), _ in test_data
    ]
    y_true = [response for (_, _), response in test_data]
    return roc_auc_score(y_true, y_score)

def pairwise_disagreement_error(true_rank, estimate):
    n = len(true_rank)
    assert(len(estimate) == n)
    pred_pos = dict([(item, i) for i, item in enumerate(estimate)])
    error = 0.
    for i in range(n-1):
        for j in range(i+1, n):
            pos_i = pred_pos[true_rank[i]]
            pos_j = pred_pos[true_rank[j]]
            
            if pos_i > pos_j:
                error += 1  
    return error/(n*(n-1)/2)

def top_k_accuracy(true_rank, estimate, K):
    # Assume that the items are sorted from most popular to least popular
    true_top_k = true_rank[:K]
    estimate_top_k = estimate[:K]
    return float(len(np.intersect1d(true_top_k, estimate_top_k)))/K