#!/usr/bin/env python3
import numpy as np
import time
import scipy.stats as s
import csv

path = '/home/ubuntu/abc_sir/py_code/abc_smc_data/10k_0.2_0.03/'

# ---------------------------
# Simulator: SIR model
# ---------------------------
def simulator(beta, gamma, ndays, N, I0=10):
    """
    Simulate the SIR model over 'ndays' days using Euler integration.
    Starting with I0 infected, the rest susceptible, and 0 removed.
    Enforces S+I+R == N at every step.
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
        # New infections and recoveries using binomial draws
        dI[t] = np.random.binomial(int(S[t-1]), 1 - np.exp(-beta * frac_infected[t-1]))
        dR[t] = np.random.binomial(int(I[t-1]), gamma)
        
        S[t] = S[t-1] - dI[t]
        I[t] = I[t-1] + dI[t] - dR[t]
        R[t] = R[t-1] + dR[t]
        frac_infected[t] = I[t] / N
        
        # Enforce S+I+R == N
        if S[t] + I[t] + R[t] != N:
            raise ValueError('The sum of S, I, and R should equal the total population size.')
    return S, I, R

# ---------------------------
# Prior sampler
# ---------------------------
def prior(exponent_scale=0.2, beta_a=0.03, beta_b=1.):
        """
        Sample parameters from the prior.
        beta ~ Exponential(1) to get beta in (0, inf)
        gamma ~ Uniform(0,1)
        """
        beta = s.expon.rvs(scale=exponent_scale, random_state=np.random)
        gamma = s.beta.rvs(a=beta_a, b=beta_b, random_state=np.random)
        return beta, gamma

# ---------------------------
# Distance function
# ---------------------------
def distance(N, observed, simulated):
    """
    Compute the distance between observed and simulated data.
    Uses the normalized L2 norm of differences for S, I, and R.
    
    Parameters:
        N        : Total population (normalization factor)
        observed : Tuple (S_obs, I_obs, R_obs)
        simulated: Tuple (S_sim, I_sim, R_sim)
        
    Returns:
        d_S, d_I, d_R (floats)
    """
    obs_S, obs_I, obs_R = observed
    sim_S, sim_I, sim_R = simulated
    
    d_S = np.linalg.norm(sim_S - obs_S) / N
    d_I = np.linalg.norm(sim_I - obs_I) / N
    d_R = np.linalg.norm(sim_R - obs_R) / N
    
    return d_S, d_I, d_R

# ---------------------------
# Perturbation kernel for ABC SMC
# ---------------------------
def perturb_sample(accepted_samples, std_beta, std_gamma):
    """
    Pick a random accepted sample and perturb it.
    Ensures β remains positive and γ remains in [0, 1].
    """
    idx = np.random.randint(len(accepted_samples))
    beta, gamma = accepted_samples[idx]
    
    while True:
        new_beta = beta + np.random.normal(0, std_beta)
        new_gamma = gamma + np.random.normal(0, std_gamma)
        if new_beta > 0 and 0 <= new_gamma <= 1:
            break
#     # Enforce parameter bounds
#     new_beta = new_beta if new_beta > 0 else 1e-3
#     new_gamma = min(max(new_gamma, 0), 1)
    return new_beta, new_gamma

# ---------------------------
# Main ABC SMC routine
# ---------------------------
def run_abc(n_particles, n_generations, ndays, N, I0, observed):
    """
    Run a simple sequential ABC sampler.
    
    For generation 0, particles are drawn from the prior.
    For subsequent generations, particles are drawn by perturbing
    samples from the previous generation.
    
    After each generation, the tolerance for each metric (S, I, R) is
    updated to be the 75th percentile of the accepted distances.
    
    A particle is accepted if its simulated distances are each less than or equal
    to the corresponding tolerance.
    
    Returns:
        A list (per generation) of accepted parameter tuples (beta, gamma).
    """
    # Helper function to save accepted parameters to CSV
    def save_to_csv(params, generation):
        with open(f'{path}accepted_params_gen_{generation}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["beta", "gamma"])
            for p in params:
                writer.writerow(p)
    
    # Helper function to save generation stats (execution time and tolerance) to CSV
    def save_generation_stats(stats):
        # path = '/home/ubuntu/abc_sir/py_code/abc_smc_data/'
        with open(f'{path}generation_stats.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["generation", "exec_time", "tol_S", "tol_I", "tol_R"])
            for row in stats:
                writer.writerow(row)
    
    # Initial tolerance: set very high so that all particles are accepted in generation 0.
    tolerance = [np.inf, np.inf, np.inf]
    prev_accepted = None
    accepted_params_all = []
    generation_stats = []  # list to store generation stats
    
    for gen in range(n_generations):
        start_time = time.time()
        accepted_params = []
        accepted_distances = []
        
        if gen == 0:
            # Sample directly from the prior
            sample_func = prior
        else:
            # Estimate perturbation scales from previous generation
            params_array = np.array(prev_accepted)
            std_beta = np.std(params_array[:, 0])
            std_gamma = np.std(params_array[:, 1])
            # Ensure nonzero standard deviations
            std_beta = std_beta if std_beta > 0 else 0.1
            std_gamma = std_gamma if std_gamma > 0 else 0.1
            
            # Use a lambda that perturbs a previous accepted sample
            sample_func = lambda: perturb_sample(prev_accepted, std_beta, std_gamma)
        
        # Continue sampling until we have n_particles accepted for this generation
        while len(accepted_params) < n_particles:
            beta, gamma = sample_func()
            sim_data = simulator(beta, gamma, ndays, N, I0)
            dS, dI, dR = distance(N, observed, sim_data)
            # Accept if distances are less than or equal to current tolerances for all metrics
            if dS <= tolerance[0] and dI <= tolerance[1] and dR <= tolerance[2]:
                accepted_params.append((beta, gamma))
                accepted_distances.append((dS, dI, dR))
        
        # Save the accepted parameters for the current generation (beta, gamma) to CSV
        save_to_csv(accepted_params, gen)
        
        accepted_distances = np.array(accepted_distances)
        # Update tolerances for next generation as the 75th percentile for each metric
        new_tol = [np.percentile(accepted_distances[:, 0], 75),
                   np.percentile(accepted_distances[:, 1], 75),
                   np.percentile(accepted_distances[:, 2], 75)]
        print(f"Generation {gen} completed. New tolerance set to: {new_tol}")
        
        prev_accepted = accepted_params
        accepted_params_all.append(accepted_params)
        tolerance = new_tol
        
        exec_time = time.time() - start_time
        print(f"Generation {gen} execution time: {exec_time:.2f} seconds")
        generation_stats.append([gen, exec_time, new_tol[0], new_tol[1], new_tol[2]])
    
    save_generation_stats(generation_stats)
    return accepted_params_all

# ---------------------------
# Main driver
# ---------------------------
def main():
    np.random.seed(9725)
    # Simulation settings
    ndays = 600      # number of days to simulate
    N = 1000        # total population
    I0 = 10         # initial number of infected
    
    # True parameters used to generate "observed" data
    true_beta = 0.2
    true_gamma = 0.03
    observed = simulator(true_beta, true_gamma, ndays, N, I0)
    
    # ABC SMC settings
    n_particles = 10000   # number of particles per generation
    n_generations = 16   # number of generations
    
    print("Starting ABC SMC...")
    results = run_abc(n_particles, n_generations, ndays, N, I0, observed)
    
#     print("\nFinal accepted parameters (last generation):")
#     for i, (beta, gamma) in enumerate(results[-1]):
#         print(f"Particle {i+1}: beta = {beta:.4f}, gamma = {gamma:.4f}")

if __name__ == '__main__':
    main()
