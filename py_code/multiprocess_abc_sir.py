import multiprocessing
# ...existing code...

# Add the following worker function at module level:
def abc_worker(args):
    """
    Worker function to run a single accepted sample loop.
    """
    exponent_scale, beta_a, beta_b, simulator_func, distance_func, ndays, N, epsilon, observed = args
    total_trials = 0
    import scipy.stats as s  # local import for multiprocessing safety
    while True:
        total_trials += 1
        beta = s.expon.rvs(scale=exponent_scale)
        gamma = s.beta.rvs(a=beta_a, b=beta_b)
        S, I, R = simulator_func(beta, gamma, ndays, N)
        simulated = np.array([S, I, R], ndmin=2)
        d_S, d_I, d_R = distance_func(N, observed, simulated)
        if (d_S <= epsilon) and (d_I <= epsilon) and (d_R <= epsilon):
            return beta, gamma, total_trials

# In the ABCSIR class, update the run method:
class ABCSIR:
    # ...existing code...
    
    def run(self):
        """
        Run the ABC sampling algorithm in parallel using multiprocessing.
        """
        args_list = [(self.exponent_scale, self.beta_a, self.beta_b,
                      self.simulator_func, self.distance_func,
                      self.ndays, self.N, self.epsilon_fixed, self.observed)
                     for _ in range(self.n_iterations)]
        
        with multiprocessing.Pool(processes=3) as pool:
            results = pool.map(abc_worker, args_list)
        
        total_trials_array = [res[2] for res in results]
        self.accepted_betas = [res[0] for res in results]
        self.accepted_gammas = [res[1] for res in results]
        self.total_proposals = sum(total_trials_array)
        avg_trials = np.mean(total_trials_array)
        self.acceptance_rate = self.n_iterations / float(self.total_proposals)
        print(f'Acceptance rate: {self.acceptance_rate:.3f}', f'Average number of trials: {avg_trials:.3f}')