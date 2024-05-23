import numpy as np
INVALID_RESPONSE = -99999

def partition_data(A, p=0.8, seed=None ):
    m, n = A.shape[0], A.shape[1]
    np.random.seed(seed)
    A_train = np.copy(A)
    A_test = np.copy(A)
    
    # For each row, randomly removing some of the students scores
    # Or equivalently randomly select a subset of the students' performance
    for i in range(m):
        responses = A[i, :]
        responses_idx = np.where(responses != INVALID_RESPONSE)
        num_response = len(responses_idx)
        subset_train = np.random.choice(num_response, size=(int(p * num_response),), replace=False)
        subset_test = np.delete(responses_idx, subset_train)
        
        A_train[i, subset_test] = INVALID_RESPONSE
        A_test[i, subset_train] = INVALID_RESPONSE
    
    return A_train, A_test