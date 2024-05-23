import numpy as np
import itertools

class Embedding:
    def __init__(self, n):
        self.n = n

    def fit(self, ranks):
        embeddings = []
        for rank in ranks:
            embeddings.append(self.embed(rank))
        return np.array(embeddings)

    def embed(self, rank):
        raise NotImplementedError


class LehmerCode(Embedding):
    def __init__(self, n):
        super().__init__(n)

    def embed(self, rank):
        v = np.zeros(self.n)
        seen = set()
        for i in rank:
            seen.add(i)
            for j in range(i):
                if j not in seen:
                    v[j] += 1 # Add 1 to c(j) for all j > i
        return [v[i] for i in rank]


class PositionalEmbedding(Embedding):
    def __init__(self, n):
        super().__init__(n)

    def embed(self, rank):
        v = np.zeros((self.n,))
        for i, item in enumerate(rank):
            v[item] = i
        return v


class PairwiseEmbedding(Embedding):
    def __init__(self, n, convolution=False, approximate=True, num_mc_samples=1000):
        super().__init__(n)
        self.convolution = convolution # Whether to compute the kernel mean embedding over all consistent full permutations
        self.approximate = approximate # Approximate for the case of partial rankings, only important if convolution is True
        self.num_mc_samples = num_mc_samples # Only used when both convolution and approximate are true

    def embed(self, rank):
        if len(rank) == self.n or not self.convolution:
            return self.embed_full(rank)
        else:
            return self.embed_partial(rank)
    
    def embed_full(self, rank):
        v = np.zeros((int(self.n * (self.n-1) / 2),))
        for i in range(len(rank)-1):
            for j in range(i+1, len(rank)):
                # item i beats item j
                itemi = rank[i]
                itemj = rank[j]

                # By default we encode the pair (i, j) for i < j
                if itemi < itemj:
                    # Mapping from (itemi, itemj) pair to 1d index in the embedding vector
                    idx = self.n * itemi - int((itemi + 1) * (itemi) / 2) + (itemj - itemi - 1)
                    # v[idx] = 1./2 
                    v[idx] = 1.
                else:
                    idx = self.n * itemj - int((itemj + 1) * (itemj) / 2) + (itemi - itemj - 1)
                    # v[idx] = -1./2

        return v
    
    def embed_partial(self, rank):
        ns = len(rank)
        rank = np.array(rank, dtype=int)
        other_items = np.array([item for item in range(self.n) if item not in rank])
        if self.approximate:
            # Approximate the convolution
            sampled_full_rankings = np.zeros((self.num_mc_samples, self.n), dtype=int)
            # Within each ranking, turn on exactly ns bits
            # Place the items in rank in these bits
            # For the rest of the items, place them uniformly at random.
            for i in range(self.num_mc_samples):
                indices = sorted(np.random.choice(self.n, ns, False))
                sampled_full_rankings[i, indices] = rank
                # Place the remaining items at random
                other_indices = np.array([x for x in range(self.n) if x not in indices])
                np.random.shuffle(other_items) # Generate a random permutation
                sampled_full_rankings[i, other_indices] = other_items
            
            # Compute the average embedding of these permutations
            mean_embedding = np.zeros((int(self.n * (self.n-1)/2), ))
            for sampled_ranking in sampled_full_rankings:
                mean_embedding += self.embed_full(sampled_ranking)
            return mean_embedding/ self.num_mc_samples
            
        else:
            # Too time consuming even for n ~ 10
            # Compute in full over all permutations satisfying the partial order in rank
            mean_embedding = np.zeros((int(self.n * (self.n-1)/2), ))
            count = 0
            full_rank = np.zeros((self.n,), dtype=int)
            
            for indices in itertools.combinations(range(self.n), ns):    
                # print(indices)
                indices = np.array(indices)
                full_rank[indices] = rank
                other_indices =  np.array([x for x in range(self.n) if x not in indices])

                for permutation in itertools.permutations(other_items):
                    count += 1
                    full_rank[other_indices] = permutation
                    mean_embedding += self.embed_full(full_rank)
                
            # Generate all possible ns combination of positions for the ranked item
            # For each combination of positions, generate all possible permutations
            # for the remaining items, compute the average (need to be normalized properly) or sum
            return mean_embedding/count
        

class TopKEmbedding(Embedding):
    def __init__(self, n, K):
        super().__init__(n)
        self.K = K

    def embed(self, rank):
        v = np.zeros((self.n, ))
        for i in rank[:self.K]:
            v[i] = 1
        return v


class DisjointPairEmbedding(Embedding):
    def __init__(self, n):
        super().__init__(n)
        # For now, consider the the simple case where n is even

    def embed(self, rank):
        v = np.zeros((int(self.n/2),))
        for i in rank:
            if i % 2 == 0: # If even
                if v[int(i/2)] == 0:
                    v[int(i/2)] = 1
            else:
                if v[int(i/2)] == 0:
                    v[int(i/2)] = -1
        return v


