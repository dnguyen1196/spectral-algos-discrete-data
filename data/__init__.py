import numpy as np
from itertools import permutations
from scipy.special import softmax, logsumexp
from pyrsistent import v


def validate_and_normalize(U, alpha=None):
    if not np.all(np.isfinite(U)):
        U = np.zeros((U.shape))
    if alpha is not None:
        if not np.all(np.isfinite(alpha)):
            alpha = np.ones(alpha.shape) * 1./len(alpha)
        alpha /= np.sum(alpha)

    U = U - np.mean(U, 1)[:, np.newaxis]
    if alpha is not None:
        return U, alpha
    return U


def top_K_accuracy(rank, rank_est, K):
    return len(np.intersect1d(rank[:K], rank_est[:K]))/K


def ell2(U, Uhat):
    return np.linalg.norm((Uhat-np.mean(Uhat)) - (U-np.mean(U)))

def pairwise_loglikelihood(P, U, F):
    n = len(U)
    Delta = np.outer(np.ones((n,)), U) - np.outer(U, np.ones((n,))) # Delta[i, j] = F(Uj - Ui) = Phat(j beats i)
    return np.sum(P * np.log(F(Delta)))


def log_likelihood_pl(rankings, U):
    pi = softmax(U)
    log_likelihoods = []
    for ranking in rankings:
        pi_sigma = np.array([pi[i] for i in ranking])
        # Use cumsum here
        sum_pi = np.cumsum(pi_sigma[::-1])[::-1]
        log_lik = np.log(pi_sigma/sum_pi)
        log_lik = np.sum(log_lik[:-1])
        log_likelihoods.append(log_lik)
    res = np.array(log_likelihoods)
    return res

def estimate_log_likelihood_pl(rankings, U_all, alpha):
    """Given a learned model, estimate PL log likelihood on a set of rankings

    :param rankings: _description_
    :type rankings: _type_
    :param U_all: _description_
    :type U_all: _type_
    :param noise: _description_
    :type noise: _type_
    """
    alpha = np.array(alpha)
    alpha /= np.sum(alpha)
    m = len(rankings)
    K = len(alpha)
    log_likelihoods = np.zeros((m, K))
    for k in range(K):
        log_likelihoods[:, k] = log_likelihood_pl(rankings, U_all[k]) + np.log(alpha[k])

    log_likelihoods = logsumexp(log_likelihoods, 1)
    assert(np.all(log_likelihoods < 0))
    assert(log_likelihoods.shape == (m,))
    return np.mean(log_likelihoods)


def log_likelihood(rankings, U, alpha):
    U, alpha = validate_and_normalize(U, alpha)
    # Compute log-likelihood on some heldout rankings
    return estimate_log_likelihood_pl(rankings, U, alpha)


def ell2_mixture(U, Uhat):
    U = validate_and_normalize(U)
    Uhat = validate_and_normalize(Uhat)
    
    K = len(U)
    assert(U.shape == Uhat.shape)

    def modified_eucl_dist(x, y):
        assert(x.shape == y.shape)
        return np.linalg.norm(x-y)

    best_alignment = list(range(K))
    lowest_dist = np.inf
    for perm in list(permutations(range(K))): # Try every possible permutation
        dist = np.sum([modified_eucl_dist(U[i,:], Uhat[k,:])**2 for i, k in enumerate(perm)]) # Compute overall error

        if dist < lowest_dist: # Pick the best alignment (wih lowest error)
            lowest_dist = dist
            best_alignment = perm
    
    alignment = dict([(i, k) for i, k in enumerate(best_alignment)])
    Uhat_aligned = np.zeros_like(Uhat)
    for (i, j) in alignment.items():
        Uhat_aligned[i, :] = Uhat[j, :]
    ell2 = np.array([modified_eucl_dist(U[k, :], Uhat_aligned[k,:]) for k in range(K)])
    return np.linalg.norm(ell2)

import numpy as np
from sklearn.metrics import confusion_matrix
from itertools import permutations


# Copied from https://github.com/scikit-learn/scikit-learn/blob/0.22.X/sklearn/utils/linear_assignment_.py
# The current version of linear_sum_assignment is not the same

def linear_assignment(X):
    """Solve the linear assignment problem using the Hungarian algorithm.
    The problem is also known as maximum weight matching in bipartite graphs.
    The method is also known as the Munkres or Kuhn-Munkres algorithm.
    Parameters
    ----------
    X : array
        The cost matrix of the bipartite graph
    Returns
    -------
    indices : array
        The pairs of (row, col) indices in the original array giving
        the original ordering.
    References
    ----------
    1. http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html
    2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
       *Naval Research Logistics Quarterly*, 2:83-97, 1955.
    3. Harold W. Kuhn. Variants of the Hungarian method for assignment
       problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.
    4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
       *Journal of the Society of Industrial and Applied Mathematics*,
       5(1):32-38, March, 1957.
    5. https://en.wikipedia.org/wiki/Hungarian_algorithm
    """
    indices = _hungarian(X).tolist()
    indices.sort()
    # Re-force dtype to ints in case of empty list
    indices = np.array(indices, dtype=int)
    # Make sure the array is 2D with 2 columns.
    # This is needed when dealing with an empty list
    indices.shape = (-1, 2)
    return indices


class _HungarianState:
    """State of one execution of the Hungarian algorithm.
    Parameters
    ----------
    cost_matrix : 2D matrix
        The cost matrix. Does not have to be square.
    """

    def __init__(self, cost_matrix):
        cost_matrix = np.atleast_2d(cost_matrix)

        # If there are more rows (n) than columns (m), then the algorithm
        # will not be able to work correctly. Therefore, we
        # transpose the cost function when needed. Just have to
        # remember to swap the result columns back later.
        transposed = (cost_matrix.shape[1] < cost_matrix.shape[0])
        if transposed:
            self.C = (cost_matrix.T).copy()
        else:
            self.C = cost_matrix.copy()
        self.transposed = transposed

        # At this point, m >= n.
        n, m = self.C.shape
        self.row_uncovered = np.ones(n, dtype=np.bool)
        self.col_uncovered = np.ones(m, dtype=np.bool)
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = np.zeros((n + m, 2), dtype=int)
        self.marked = np.zeros((n, m), dtype=int)

    def _clear_covers(self):
        """Clear all covered matrix cells"""
        self.row_uncovered[:] = True
        self.col_uncovered[:] = True


def _hungarian(cost_matrix):
    """The Hungarian algorithm.
    Calculate the Munkres solution to the classical assignment problem and
    return the indices for the lowest-cost pairings.
    Parameters
    ----------
    cost_matrix : 2D matrix
        The cost matrix. Does not have to be square.
    Returns
    -------
    indices : 2D array of indices
        The pairs of (row, col) indices in the original array giving
        the original ordering.
    """
    state = _HungarianState(cost_matrix)

    # No need to bother with assignments if one of the dimensions
    # of the cost matrix is zero-length.
    step = None if 0 in cost_matrix.shape else _step1

    while step is not None:
        step = step(state)

    # Look for the starred columns
    results = np.array(np.where(state.marked == 1)).T

    # We need to swap the columns because we originally
    # did a transpose on the input cost matrix.
    if state.transposed:
        results = results[:, ::-1]

    return results


# Individual steps of the algorithm follow, as a state machine: they return
# the next step to be taken (function to be called), if any.

def _step1(state):
    """Steps 1 and 2 in the Wikipedia page."""

    # Step1: For each row of the matrix, find the smallest element and
    # subtract it from every element in its row.
    state.C -= state.C.min(axis=1)[:, np.newaxis]
    # Step2: Find a zero (Z) in the resulting matrix. If there is no
    # starred zero in its row or column, star Z. Repeat for each element
    # in the matrix.
    for i, j in zip(*np.where(state.C == 0)):
        if state.col_uncovered[j] and state.row_uncovered[i]:
            state.marked[i, j] = 1
            state.col_uncovered[j] = False
            state.row_uncovered[i] = False

    state._clear_covers()
    return _step3


def _step3(state):
    """
    Cover each column containing a starred zero. If n columns are covered,
    the starred zeros describe a complete set of unique assignments.
    In this case, Go to DONE, otherwise, Go to Step 4.
    """
    marked = (state.marked == 1)
    state.col_uncovered[np.any(marked, axis=0)] = False

    if marked.sum() < state.C.shape[0]:
        return _step4


def _step4(state):
    """
    Find a noncovered zero and prime it. If there is no starred zero
    in the row containing this primed zero, Go to Step 5. Otherwise,
    cover this row and uncover the column containing the starred
    zero. Continue in this manner until there are no uncovered zeros
    left. Save the smallest uncovered value and Go to Step 6.
    """
    # We convert to int as numpy operations are faster on int
    C = (state.C == 0).astype(np.int)
    covered_C = C * state.row_uncovered[:, np.newaxis]
    covered_C *= state.col_uncovered.astype(dtype=np.int, copy=False)
    n = state.C.shape[0]
    m = state.C.shape[1]
    while True:
        # Find an uncovered zero
        row, col = np.unravel_index(np.argmax(covered_C), (n, m))
        if covered_C[row, col] == 0:
            return _step6
        else:
            state.marked[row, col] = 2
            # Find the first starred element in the row
            star_col = np.argmax(state.marked[row] == 1)
            if not state.marked[row, star_col] == 1:
                # Could not find one
                state.Z0_r = row
                state.Z0_c = col
                return _step5
            else:
                col = star_col
                state.row_uncovered[row] = False
                state.col_uncovered[col] = True
                covered_C[:, col] = C[:, col] * (
                    state.row_uncovered.astype(dtype=np.int, copy=False))
                covered_C[row] = 0


def _step5(state):
    """
    Construct a series of alternating primed and starred zeros as follows.
    Let Z0 represent the uncovered primed zero found in Step 4.
    Let Z1 denote the starred zero in the column of Z0 (if any).
    Let Z2 denote the primed zero in the row of Z1 (there will always be one).
    Continue until the series terminates at a primed zero that has no starred
    zero in its column. Unstar each starred zero of the series, star each
    primed zero of the series, erase all primes and uncover every line in the
    matrix. Return to Step 3
    """
    count = 0
    path = state.path
    path[count, 0] = state.Z0_r
    path[count, 1] = state.Z0_c

    while True:
        # Find the first starred element in the col defined by
        # the path.
        row = np.argmax(state.marked[:, path[count, 1]] == 1)
        if not state.marked[row, path[count, 1]] == 1:
            # Could not find one
            break
        else:
            count += 1
            path[count, 0] = row
            path[count, 1] = path[count - 1, 1]

        # Find the first prime element in the row defined by the
        # first path step
        col = np.argmax(state.marked[path[count, 0]] == 2)
        if state.marked[row, col] != 2:
            col = -1
        count += 1
        path[count, 0] = path[count - 1, 0]
        path[count, 1] = col

    # Convert paths
    for i in range(count + 1):
        if state.marked[path[i, 0], path[i, 1]] == 1:
            state.marked[path[i, 0], path[i, 1]] = 0
        else:
            state.marked[path[i, 0], path[i, 1]] = 1

    state._clear_covers()
    # Erase all prime markings
    state.marked[state.marked == 2] = 0
    return _step3


def _step6(state):
    """
    Add the value found in Step 4 to every element of each covered row,
    and subtract it from every element of each uncovered column.
    Return to Step 4 without altering any stars, primes, or covered lines.
    """
    # the smallest uncovered value in the matrix
    if np.any(state.row_uncovered) and np.any(state.col_uncovered):
        minval = np.min(state.C[state.row_uncovered], axis=0)
        minval = np.min(minval[state.col_uncovered])
        state.C[np.logical_not(state.row_uncovered)] += minval
        state.C[:, state.col_uncovered] -= minval
    return _step4

# Taken from https://coclust.readthedocs.io/en/v0.2.1/_modules/coclust/evaluation/external.html#accuracy

def clustering_accuracy(true_row_labels, predicted_row_labels):
    """Get the best accuracy.

    Parameters
    ----------
    true_row_labels: array-like
        The true row labels, given as external information
    predicted_row_labels: array-like
        The row labels predicted by the model

    Returns
    -------
    float
        Best value of accuracy
    """

    cm = confusion_matrix(true_row_labels, predicted_row_labels)
    indexes = linear_assignment(_make_cost_m(cm))
    total = 0
    for row, column in indexes:
        value = cm[row][column]
        total += value

    return (total * 1. / np.sum(cm))



def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)
