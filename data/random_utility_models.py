import numpy as np


def normal_noise(k, s):
    return np.random.normal(0, s, size=k)


def gumbel_noise(k, s):
    return np.random.gumbel(0, s, size=k)


def exponential_noise(k, s):
    return np.random.exponential(s, size=k)


def normal_std(k):
    return normal_noise(k, 1.)


def gumbel_std(k):
    return gumbel_noise(k, 1.)


def exp_std(k):
    return exponential_noise(k, 1.)


def normal_half(k):
    return normal_noise(k, 1./2)


def compute_mean_pairwise(U, F):
    n = len(U)
    v = np.zeros((int(n * (n-1) / 2),))
    for i in range(len(U)-1):
        for j in range(i+1, len(U)):
            # item i beats item j
            Ui = U[i]
            Uj = U[j]

            # Mapping from (i, j) edge to index in the embedding
            idx = n * i - int((i + 1) * (i) / 2) + (j - i - 1)
            
            # 1./2 Pij - 1./2 Pji
            v[idx] = F(Ui - Uj) * 1./2 - 1./2 * F(Uj - Ui)

    return v