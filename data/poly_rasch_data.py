import numpy as np


def generate_polytomous_rasch(thetas, betas):
    """
    Generate synthetic data according to the PCM model.
    Args:
        thetas: A (n_users,) vector representing the students' abilities
        betas: A (n_tests, K) matrox representing the tests' difficulties
    Returns:
        A matrix X with shape (n_users, n_tests) generated per the PCM.
        
    """
    
    n = len(thetas)
    m, L = betas.shape # Number of items, number of levels
    results = np.zeros((n, m))
    # Compute Pnil for all (student, test, level) tuple
    # For each (student)
    for test_id, beta in enumerate(betas):
        # Go through each question to generate the students' responses.
        # Ignoring the test index for now
        
        # ability_ms_diffty[i, l] = betai - theta_l
        # This has shape (n, L)
        ability_ms_diff = thetas[:, None] - beta.reshape((1, L))
        
        # Pnk = logit(beta_n - theta_k) = prob(student n overcomes threshold k)
        p = logistic_sigmoid(ability_ms_diff)
        # Qnik = 1 - Pnik
        q = 1. - p # Complementary fail probability
        
        # Compute the running Pnik product
        # P(Xni = l) = passes all level from 1 to l, but fails from l+1 to L
        probs = np.zeros((n, L+1))
        for l in range(L+1):
            # For the two extreme cases where l = 0, or L
            if l == 0:
                probs[:, l] = np.prod(q, 1)
            elif l == L:
                probs[:, l] = np.prod(p, 1)
            else:
                # p[:, l] = [Pn1, Pn2, ... , Pnl]
                # q[:, l:] = [Qnl+1,..., QnL]
                probs[:, l] = np.prod(p[:, :l], 1) * np.prod(q[:, l:], 1)
                
        # Normalize by row to obtain grade probability
        row_sum = np.sum(probs, 1)
        # Draw student results from categorical distribution
        probs /= row_sum[:, None] # Should have shape (n, L+1)
        # Draw performance for each student individually
        for student_id in range(n):
            pi_student = probs[student_id, :]
            results[student_id, test_id] = np.random.choice(L+1, p=pi_student)
    
    return results


