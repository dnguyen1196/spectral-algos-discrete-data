import numpy as np
from scipy.special import softmax


class BayesianEMGS:
    def __init__(self, n, G, prior_c=None, prior_d=None, prior_alpha=None):
        self.n = n
        self.G = G
        self.prior_c = prior_c if prior_c is not None else 1
        self.prior_d = prior_d if prior_d is not None else 0
        self.prior_alpha = prior_alpha if prior_alpha is not None else 1
        self.theta = None
        
    def fit(self, rankings, theta_init=None):
        u, delta = self.collect_statistics(rankings)
        p, w = self.map_estimate(u, delta, rankings, softmax(theta_init, 1))
        self.theta = np.log(p)
        self.theta = self.theta - np.mean(self.theta)
        return self.theta
    
    def sample_posterior(self, rankings, n_samples, u, delta, p, w):
        # Draw samples from the posterior distribution over the parameters
        # Get an initial z to get the process started 
        m = len(rankings)
        pz = np.zeros((m, self.G))
        for g in range(self.G):
            pz[:, g] = self.estimate_log_likelihood_pl(rankings, p[g, :]) + np.log(w[g])
        z = np.zeros((m,))
        for s in range(m):
            z[s] = int(np.argmax(pz[s,:]))
        y = None
        posterior_samples = []
        # Repeat the process
        for _ in range(n_samples):
            # Draw y
            y = self.draw_y(rankings, y, z, p, w, u, delta)
            # Then draw z
            z = self.draw_z(rankings, y, z, p, w, u, delta)
            # Then draw p
            p = self.draw_p(rankings, y, z, p, w, u, delta)
            # Then draw w (skip this)
            w = self.draw_w(rankings, y, z, p, w, u, delta)
            
            posterior_samples.append((y, z, p, w))
            
        return posterior_samples
    
    def fit_then_sample_estimate(self, rankings, theta_init=None, n_samples=1000):
        u, delta = self.collect_statistics(rankings)
        p, w = self.map_estimate(u, delta, rankings, softmax(theta_init, 1))
        posterior_samples = self.sample_posterior(rankings, n_samples, u, delta, p, w)
        
        all_p = [np.log(p) - np.mean(np.log(p), 1)[:, np.newaxis] for (_, _, p, _) in posterior_samples]
        all_p = np.array(all_p)
        mean_p = np.mean(all_p, 0)
        
        all_alpha = [w for (_, _, _, w ) in posterior_samples]
        all_alpha = np.array(all_alpha)
        alpha = np.mean(all_alpha, 0)
        alpha /= np.sum(alpha)
        assert(mean_p.shape == (self.G, self.n))
        return mean_p, alpha
    
    def draw_y(self, rankings, y, z, p, w, u, delta):
        # Each yst is sampled independently from the gamma distribution
        m = len(rankings)
        y_sampled = np.zeros((m, self.n-1))
        
        for s in range(m):
            zs = int(z[s])
            p_in_ranking = [p[zs, i] for i in rankings[s]]
            cum_sum_p = (np.cumsum(p_in_ranking[::-1])[::-1])[:-1] # Ignore the weight of the last-ranked item
            for t in range(self.n - 1):
                # The rate is the sum of the weight of the item at position t-1 and worse
                rate = cum_sum_p[t]
                y_sampled[s, t] = np.random.exponential(1./rate)
        return y_sampled
    
    def draw_z(self, rankings, y, z, p, w, u, delta):
        # Draw z from a multinomial distribution
        m = len(rankings)
        z_sampled = np.zeros((m,))
        for s in range(m):
            pz = []
            for g in range(self.G):
                pzg = w[g]
                for i in range(self.n):
                    sum_over_t = delta[s, :, i] * y[s,:]
                    assert(sum_over_t.shape == (self.n-1,))
                    pzg *= np.power(p[g, i], u[s, i]) * np.exp(-p[g, i] * np.sum(sum_over_t))
                pz.append(pzg)
                
            pz = np.array(pz)
            pz /= np.sum(pz)
            z_sampled[s] = np.random.choice(self.G, p=pz)
        return np.array(z_sampled)
    
    def draw_p(self, rankings, y, z, p, w, u, delta):
        # Draw p from a Gamma distribution
        p_sampled = np.zeros((self.G, self.n))
        for g in range(self.G):
            for i in range(self.n):
                shape_pgi = self.prior_c + np.sum(np.where(z == g, u[:, i], 0))
                rate_pgi = self.prior_d + np.sum(np.where(z == g, np.sum(delta[:, :, i] * y, axis=1), 0))
                p_sampled[g, i] = np.random.gamma(shape_pgi, 1./rate_pgi)
        return p_sampled
    
    def draw_w(self, rankings, y, z, p, w, u, delta):
        # Draw from the dirichlet distribution with parameters
        # beta_g = alpha g - 1 + sum sg
        beta = np.zeros((self.G,))
        unique, counts = np.unique(z, return_counts=True)
        z_dict = dict(zip(unique, counts))
        for g in range(self.G):
            beta[g] = self.prior_alpha + z_dict[g] if g in z_dict else self.prior_alpha
            
        w = np.random.dirichlet(beta)
        return w
    
    def map_estimate(self, u, delta, rankings, p_init, max_iters=100, eps=1e-6):
        # Run the EM algorithm till convergence
        m = len(u)
        # Start from the uniform distribution
        w = np.ones((self.G)) * 1./self.G
        p = p_init
        
        for _ in range(max_iters):
            # Do the E-step
            z = np.zeros((m, self.G))
            for g in range(self.G):
                z[:, g] = self.estimate_log_likelihood_pl(rankings, p[g, :]) + np.log(w[g])
            z = softmax(z, 1)

            # Then do the M-step
            # Update p
            p_next = np.zeros_like(p)
            
            for g in range(self.G):
                delta_sti_pgi_sum = np.zeros((m, self.n-1)) # Shape (m, n-1)
                
                # Pre-compute some quantity
                for s in range(m):
                    for t in range(self.n-1):
                        delta_sti_pgi_sum[s, t] = np.sum(delta[s, t, :] * p[g, :])

                for i in range(self.n):
                    sum_over_t = np.sum(delta[:, :, i] / delta_sti_pgi_sum[:, :], 1) # Should have shape (m,)
                    denominator = np.sum(z[:, g] * sum_over_t) 
                    p_next[g, i] = (self.prior_c - 1 + np.sum(z[:, g] * u[:, i])) / denominator
            
            # Update w
            w_next = np.zeros_like(w)
            for g in range(self.G):
                w_next[g] = (self.prior_alpha - 1 + np.sum(z[:, g])) / (self.G * self.prior_alpha - self.G + m)
            w_next /= np.sum(w_next)
                
            # Check for convergence
            if np.sum(np.square(p_next - p)) + np.sum(np.square(w - w_next)) < eps:
                break
            w = w_next
            p = p_next
        
        return p, w
    
    def estimate_log_likelihood_pl(self, rankings, p):
        log_likelihoods = []
        for ranking in rankings:
            pi_sigma = np.array([p[i] for i in ranking])
            # Use cumsum here
            sum_pi = np.cumsum(pi_sigma[::-1])[::-1]
            log_lik = np.log(pi_sigma/sum_pi)
            log_lik = np.sum(log_lik[:-1])
            log_likelihoods.append(log_lik)
        return np.array(log_likelihoods)
    
    def collect_statistics(self, rankings):
        """
        u_{si} = 1 if item i is NOT dead last in ranking s
        delta_{sti} = 1 if item i is NOT in the top t items of ranking s
        """
        m = len(rankings)
        u = np.ones((m, self.n))
        delta = np.ones((m, self.n-1, self.n))
        
        for s, ranking in enumerate(rankings):
            u[s, ranking[-1]] = 0 # Only the last item has usi = 0 since it is not in the top n-1 items
            for t, item in enumerate(ranking[:-1]):
                delta[s, t+1:, item] = 0
        return u, delta
                    
        
        