import numpy as np
INVALID_RESPONSE = -99999


def generate_data(thetas, betas, p, missing_data_value=INVALID_RESPONSE):
    """

    thetas: is a (n_student,)-shaped vector of students' abilities
    betas: is a (n_tests, K)-shaped matrix of tests' difficulties

    """
    m = len(betas)
    n = len(thetas)
    performances = np.zeros((m, n), dtype=np.int)
    
    for i in range(m):
        betai = betas[i]
        for l in range(n):
            thetal = thetas[l]
            if np.random.rand() < p:
                if np.random.rand() < 1./(1 + np.exp(-(thetal - betai))):
                    # If the students solve the problem
                    performances[i, l] = 1
                else:
                    performances[i, l] = 0
            else:
                performances[i, l] = missing_data_value
        
    return performances