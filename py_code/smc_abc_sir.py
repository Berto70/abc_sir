from tracemalloc import start
import numpy as np
import scipy.stats as s
import matplotlib.pyplot as plt
import time as time

# ---- External utility functions ----

def simulator(beta, gamma, ndays, N, I0=10):
    """
    Simulate the SIR model over 'ndays' days.
    Use a simple Euler integration.
    Start with I0 infected, rest are susceptible, and 0 removed.
    Maintain S+I+R == N.
    """
    S = np.zeros(ndays)
    I = np.zeros(ndays)
    R = np.zeros(ndays)

    frac_infected = np.zeros(ndays)
    dI = np.zeros(ndays)
    dR = np.zeros(ndays)
    
    S[0] = N - I0
    I[0] = I0
    R[0] = 0

    frac_infected[0] = I0 / N
    
    for t in range(1, ndays):
        dI[t] = np.random.binomial(int(S[t-1]), 1 - np.exp(-beta * frac_infected[t-1]))
        dR[t] = np.random.binomial(int(I[t-1]), gamma)
        
        S[t] = S[t-1] - dI[t]
        I[t] = I[t-1] + dI[t] - dR[t]
        R[t] = R[t-1] + dR[t]

        frac_infected[t] = I[t] / N
        
        if S[t] + I[t] + R[t] != N:
            raise ValueError('S + I + R != N')
    return S, I, R

def summary(data):
    """
    Compute summary statistics from the data.
    """
    median = np.mean(data)
    spread = np.percentile(data, 90) - np.percentile(data, 10)
    return np.array([median, spread])

def distance(N, observed, simulated):
    """
    Compute a distance between the observed and simulated data.
    Uses the norm differences of the S, I, R components.
    observed and simulated are tuples: (S, I, R)
    """
    obs_S, obs_I, obs_R = observed[0], observed[1], observed[2]
    sim_S, sim_I, sim_R = simulated[0], simulated[1], simulated[2]

    d_S = np.linalg.norm(sim_S - obs_S) / N
    d_I = np.linalg.norm(sim_I - obs_I) / N
    d_R = np.linalg.norm(sim_R - obs_R) / N

    return d_S + d_I + d_R

# ---- ABC SMC Class ----
class ABCSMC:
    def __init__(self, observed, exponent_scale, beta_a, beta_b, 
                 simulator_func, distance_func,
                 ndays, N, n_generations=5, population_size=100):
        """
        observed: tuple (S, I, R) arrays for the mock dataset.
        exponent_scale, beta_a, beta_b: parameters for prior.
        simulator_func: function to simulate.
        distance_func: function to compute distance.
        ndays: total number of days.
        N: total population.
        n_generations: number of SMC generations.
        population_size: number of accepted particles in each generation.
        """
        self.observed = observed
        self.exponent_scale = exponent_scale
        self.beta_a = beta_a 
        self.beta_b = beta_b
        self.simulator_func = simulator_func
        self.distance_func = distance_func
        
        self.ndays = ndays
        self.N = N
        self.n_generations = n_generations
        self.population_size = population_size
        
        self.accepted_betas = []
        self.accepted_gammas = []
        self.epsilon_history = []
        
    def prior_func(self):
        """
        Sample parameters from the prior.
        beta ~ Exponential(scale) and gamma ~ Beta(beta_a, beta_b)
        """
        beta = s.expon.rvs(scale=self.exponent_scale)
        gamma = s.beta.rvs(a=self.beta_a, b=self.beta_b)
        return beta, gamma
        
    def run(self):
        """
        Run the ABC-SMC algorithm.
        Generation 0: sample population from the prior.
        Subsequent generations: perturb previous particles and accept if distance <= epsilon.
        The threshold epsilon is updated each generation as the 75th percentile
        of the distances of accepted particles.
        """
        # Generation 0: sample particles from the prior without rejection.
        particles = []
        distances = []
        print("Running Generation 0 ...")

        start_t = time.time()
        for i in range(self.population_size):
            beta, gamma = self.prior_func()
            S, I, R = self.simulator_func(beta, gamma, self.ndays, self.N)
            d = self.distance_func(self.N, self.observed, np.array([S, I, R], ndmin=2))
            particles.append((beta, gamma))
            distances.append(d)
        particles = np.array(particles)
        distances = np.array(distances)
        # Set initial epsilon as 75th percentile of distances.
        epsilon = np.percentile(distances, 75)
        self.epsilon_history.append(epsilon)
        print(f"Generation 0: epsilon = {epsilon:.5f}")
        
        # For subsequent generations, perturb and accept.
        for gen in range(1, self.n_generations):
            new_particles = []
            new_distances = []
            print(f"Running Generation {gen} ...")
            while len(new_particles) < self.population_size:
                # sample a particle from the previous generation uniformly.
                idx = np.random.choice(self.population_size)
                beta_old, gamma_old = particles[idx]
                # perturb parameters with a Gaussian kernel.
                new_beta = beta_old + np.random.normal(0, 0.01)
                new_gamma = gamma_old + np.random.normal(0, 0.01)
                # Enforce constraints: beta > 0 and gamma in (0,1)
                if new_beta <= 0 or new_gamma <= 0 or new_gamma > 1:
                    continue
                S, I, R = self.simulator_func(new_beta, new_gamma, self.ndays, self.N)
                d = self.distance_func(self.N, self.observed, np.array([S, I, R], ndmin=2))
                if d <= epsilon:
                    new_particles.append((new_beta, new_gamma))
                    new_distances.append(d)
            new_particles = np.array(new_particles)
            new_distances = np.array(new_distances)
            # Update epsilon as 75th percentile from this generation.
            epsilon = np.percentile(new_distances, 75)
            self.epsilon_history.append(epsilon)
            print(f"Generation {gen}: epsilon = {epsilon:.5f}")
            particles = new_particles
        # Save final accepted parameters.
        self.accepted_betas = particles[:, 0].tolist()
        self.accepted_gammas = particles[:, 1].tolist()

        end_t = time.time()
        print(f"Time taken: {end_t - start_t:.2f} seconds")
        
    def save_results(self, filename):
        """
        Save accepted beta/gamma samples and epsilon history into a .npz file.
        """
        accepted_betas = np.array(self.accepted_betas)
        accepted_gammas = np.array(self.accepted_gammas)
        epsilon_arr = np.array(self.epsilon_history)
        np.savez(filename, beta=accepted_betas, gamma=accepted_gammas, epsilon=epsilon_arr)
        print(f"Results saved to {filename}")

# ---- Plotting Functions ----

def plot_histograms(beta_samples, gamma_samples, fiducial_beta, fiducial_gamma):
    """
    Plot histograms of beta and gamma samples with fiducial values.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    axs[0].hist(beta_samples, bins=int(np.sqrt(len(beta_samples))), color='skyblue', alpha=0.7)
    axs[0].axvline(fiducial_beta, color='red', linestyle='--', label='Fiducial beta')
    axs[0].axvline(np.median(beta_samples), color='blue', linestyle='-', label='Median beta')
    axs[0].axvline(np.mean(beta_samples), color='green', linestyle='-', label='Mean beta')
    axs[0].set_title('Histogram of beta samples')
    axs[0].set_xlabel('beta')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()
    
    axs[1].hist(gamma_samples, bins=int(np.sqrt(len(gamma_samples))), color='skyblue', edgecolor='black')
    axs[1].axvline(fiducial_gamma, color='red', linestyle='--', label='Fiducial gamma')
    axs[1].axvline(np.median(gamma_samples), color='green', linestyle='-', label='Median gamma')
    axs[1].axvline(np.mean(gamma_samples), color='blue', linestyle='-', label='Mean gamma')
    axs[1].set_title('Histogram of gamma samples')
    axs[1].set_xlabel('gamma')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

def plot_trace(beta_samples, gamma_samples, fiducial_beta, fiducial_gamma):
    """
    Plot trace plots of beta and gamma samples.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    axs[0].plot(beta_samples, marker='.', linestyle='-', color='blue')
    axs[0].axhline(fiducial_beta, color='red', linestyle='--', label='Fiducial beta')
    axs[0].set_title('Trace plot for beta')
    axs[0].set_ylabel('beta')
    axs[0].legend()
    
    axs[1].plot(gamma_samples, marker='.', linestyle='-', color='green')
    axs[1].axhline(fiducial_gamma, color='red', linestyle='--', label='Fiducial gamma')
    axs[1].set_title('Trace plot for gamma')
    axs[1].set_ylabel('gamma')
    axs[1].set_xlabel('Iteration (Accepted Sample Index)')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

# ---- Example Usage ----
if __name__ == '__main__':
    np.random.seed(42)
    
    ndays = 600
    N = 1000
    
    # Create a mock dataset using fiducial parameters.
    fiducial_beta = 0.1
    fiducial_gamma = 0.01
    observed_S, observed_I, observed_R = simulator(fiducial_beta, fiducial_gamma, ndays, N)
    observed_data = np.array([observed_S, observed_I, observed_R], ndmin=2)
    
    # Initialize and run the ABC SMC algorithm.
    abc_smc = ABCSMC(observed=observed_data, exponent_scale=0.1, beta_a=1e-2, beta_b=1,
                     simulator_func=simulator, distance_func=distance,
                     ndays=ndays, N=N, n_generations=5, population_size=100)
    abc_smc.run()
    
    # Save results.
    abc_smc.save_results('/home/ubuntu/iti/project/abc_smc_results.npz')
    
    # Plotting.
    plot_histograms(abc_smc.accepted_betas, abc_smc.accepted_gammas, fiducial_beta, fiducial_gamma)
    plot_trace(abc_smc.accepted_betas, abc_smc.accepted_gammas, fiducial_beta, fiducial_gamma)