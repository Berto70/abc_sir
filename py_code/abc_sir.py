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
        # new infections and recoveries
        dI[t] = np.random.binomial(int(S[t-1]), 1 - np.exp(-beta * frac_infected[t-1]))
        dR[t] = np.random.binomial(int(I[t-1]), gamma)
        
        S[t] = S[t-1] - dI[t]
        I[t] = I[t-1] + dI[t] - dR[t]
        R[t] = R[t-1] + dR[t]

        frac_infected[t] = I[t] / N
        
        # # Correct small floating point errors: enforce S+I+R=N.
        # total = S[t] + I[t] + R[t]
        # if total != 0:
        #     factor = N/total
        #     S[t] *= factor
        #     I[t] *= factor
        #     R[t] = N - S[t] - I[t]
        if S[t] + I[t] + R[t] != N:
            raise ValueError('The sum of S, I, and R should be equal to the total population size')
    return S, I, R

def summary(data):
    """
    Compute summary statistics from the data.
    """
    median = np.mean(data)
    spread = np.percentile(data, 0.875) - np.percentile(data, 0.125)
    summary = np.array([median, spread])
    return summary

def distance(N, observed, simulated):
    """
    Compute a distance between the observed and simulated data.
    Here we use a simple sum of squared differences on the I time series.
    observed and simulated are tuples: (S, I, R)
    """
    obs_S, obs_I, obs_R = observed[0], observed[1], observed[2]
    sim_S, sim_I, sim_R = simulated[0], simulated[1], simulated[2]

    d_S = np.linalg.norm(sim_S - obs_S)/N
    d_I = np.linalg.norm(sim_I - obs_I)/N
    d_R = np.linalg.norm(sim_R - obs_R)/N

    # d_S = np.linalg.norm(summary(observed[0]) - summary(simulated[0]))/N
    # d_I = np.linalg.norm(summary(observed[1]) - summary(simulated[1]))/N
    # d_R = np.linalg.norm(summary(observed[2]) - summary(simulated[2]))/N
    
    return d_S, d_I, d_R


# ---- ABCSIR Class ----
class ABCSIR:
    def __init__(self, observed, 
                 exponent_scale, beta_a, beta_b, 
                 simulator_func, distance_func,
                 ndays, N, n_iterations=1000, epsilon_fixed=None):
        """
        observed: tuple with (S, I, R) arrays for the mock dataset.
        prior_func: function to sample from the prior.
        simulator_func: function to simulate SIR given beta, gamma.
        distance_func: function to compute distance between observed and simulated.
        epsilon_fixed: if provided, use fixed epsilon value; otherwise,
                       use 75th percentile rule.
        n_iterations: number of ABC samples to try.
        """
        self.observed = observed
        # self.prior_func = prior_func
        self.simulator_func = simulator_func
        self.distance_func = distance_func

        self.exponent_scale = exponent_scale
        self.beta_a = beta_a 
        self.beta_b = beta_b
        
        self.ndays = ndays
        self.N = N
        self.n_iterations = n_iterations
        self.epsilon_fixed = epsilon_fixed  # if None, use percentile method
        
        # Storage for accepted samples and simulation data.
        self.accepted_betas = []
        self.accepted_gammas = []
        self.accepted_S = []
        self.accepted_I = []
        self.accepted_R = []
        self.distances = []
        self.epsilon_history = []  # will be filled only if percentile mode
        
        self.total_proposals = 0

    def prior_func(self):
        """
        Sample parameters from the prior.
        beta ~ Exponential(1) to get beta in (0, inf)
        gamma ~ Uniform(0,1)
        """
        beta = s.expon.rvs(scale=self.exponent_scale)
        gamma = s.beta.rvs(a=self.beta_a, b=self.beta_b)
        return beta, gamma
        
    def run(self):
        """
        Run the ABC sampling algorithm.
        If using fixed epsilon, accept each simulation if distance <= epsilon_fixed.
        Otherwise, after collecting distances for all n_iterations proposals,
        set epsilon as the 75th percentile of distances and accept those proposals
        that are below it.
        """
        # proposals = np.zeros(())
        # distances = []
        # sims = []  # store simulation outputs (S, I, R)
        params = []  # store (beta, gamma) for each proposal

        time_in = time.time()
        
        for _ in range(self.n_iterations):
            n = 0
            n_trials = np.array([])
            while True:

                n += 1
            
            #     beta, gamma = self.prior_func()
            #     # ensure parameter domains:
            #     if beta > 0 and gamma > 0 and gamma <= 1: break
            #    # continue
                self.total_proposals += 1
                beta, gamma = self.prior_func()

                S, I, R = self.simulator_func(beta, gamma, self.ndays, self.N)
                simulated = np.array([S, I, R], ndmin=2)
                d_S, d_I, d_R = self.distance_func(self.N, self.observed, simulated)
                if _ % 100 == 0:
                    print(f'd_S: {d_S:.3f}, d_I: {d_I:.3f}, d_R: {d_R:.3f}')
                # proposals.append((beta, gamma, d_S, d_I, d_R))
                # params.append((beta, gamma))
                # distances.append((d_S, d_I, d_R))
                # sims.append((S, I, R))
            
            # distances = np.array(distances)
            
                if self.epsilon_fixed is not None:
                    epsilon = self.epsilon_fixed
                    # for (beta, gamma, d_S, d_I, d_R), (S, I, R) in zip(proposals, sims):
                    if (d_S <= epsilon) and (d_I <= epsilon) and (d_R <= epsilon):
                        n_trials = np.append(n_trials, n)
                        self.accepted_betas.append(beta)
                        self.accepted_gammas.append(gamma)
                        break
                        # self.accepted_S.append(S)
                        # self.accepted_I.append(I)
                        # self.accepted_R.append(R)
                        # self.distances.append((d_S, d_I, d_R))
                    # Save fixed epsilon for all accepted samples.
                # self.epsilon_history = np.full(len(self.accepted_betas), epsilon)
                # else:
                #     # use the 75th percentile as epsilon
                #     epsilon_S = np.percentile(distances[0], 75)
                #     epsilon_I = np.percentile(distances[1], 75)
                #     epsilon_R = np.percentile(distances[2], 75)

                #     self.epsilon_history = []  # record epsilon for each accepted proposal (same value here)
                #     for (beta, gamma, d_S, d_I, d_R), (S, I, R) in zip(proposals, sims):
                #         if d_S <= epsilon_S and d_I <= epsilon_I and d_R <= epsilon_R:
                #             self.accepted_betas.append(beta)
                #             self.accepted_gammas.append(gamma)
                #             self.accepted_S.append(S)
                #             self.accepted_I.append(I)
                #             self.accepted_R.append(R)
                #             self.distances.append((d_S, d_I, d_R))
                #             self.epsilon_history.append((epsilon_S, epsilon_I, epsilon_R))
        avg_trials = n_trials.mean()                
        self.acceptance_rate = len(self.accepted_betas) / float(self.total_proposals)
        print(f'Acceptance rate: {self.acceptance_rate:.3f}', f'Average number of trials: {avg_trials:.3f}')
        time_out = time.time()
        print(f'Time taken: {time_out - time_in:.3f} seconds')

        print('Median beta:', np.median(self.accepted_betas))
        print('Median gamma:', np.median(self.accepted_gammas))
        
    def save_results(self, filename):
        """
        Save the accepted simulation S, I, R arrays and corresponding epsilon values.
        Each key in the .npz file is a 2D array where rows correspond to an accepted sample.
        """
        # Convert lists to arrays.
        accepted_S = np.array(self.accepted_S)
        accepted_R = np.array(self.accepted_R)
        accepted_I = np.array(self.accepted_I)
        epsilon_arr = np.array(self.epsilon_history) if len(self.epsilon_history) > 0 else np.array([])
        np.savez(filename, S=accepted_S, R=accepted_R, I=accepted_I, epsilon=epsilon_arr)
        print(f'Results saved to {filename}')
        

# ---- Plotting Functions ----
def plot_histograms(beta_samples, gamma_samples, fiducial_beta, fiducial_gamma):
    """
    Two subplots: histogram for beta and histogram for gamma,
    with a vertical line for the fiducial parameter value.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    axs[0].hist(beta_samples, bins=int(np.sqrt(len(beta_samples))), color='skyblue', edgecolor='black')
    axs[0].axvline(fiducial_beta, color='red', linestyle='--', label='Fiducial beta')
    axs[0].axvline(np.median(beta_samples), color='blue', linestyle='--', label='Median beta')
    axs[0].set_title('Histogram of beta samples')
    axs[0].set_xlabel('beta')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()
    
    axs[1].hist(gamma_samples, bins=int(np.sqrt(len(gamma_samples))), color='lightgreen', edgecolor='black')
    axs[1].axvline(fiducial_gamma, color='red', linestyle='--', label='Fiducial gamma')
    axs[1].axvline(np.median(gamma_samples), color='green', linestyle='--', label='Median gamma')
    axs[1].set_title('Histogram of gamma samples')
    axs[1].set_xlabel('gamma')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

def plot_epsilon_update(epsilon_history):
    """
    Plot a line plot of epsilon updating.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(epsilon_history, marker='o')
    plt.title('Epsilon update over accepted samples')
    plt.xlabel('Accepted Sample Index')
    plt.ylabel('Epsilon Value (75th percentile)')
    plt.grid(True)
    plt.show()

def plot_trace(beta_samples, gamma_samples, fiducial_beta, fiducial_gamma):
    """
    Plot traceplots for beta and gamma samples.
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
    # Set random seed for reproducibility.
    np.random.seed(42)
    
    # Set random seed for reproducibility.


    ndays = 600
    N = 1000

    # Create a mock dataset using fiducial parameters.
    fiducial_beta = 0.1
    fiducial_gamma = 0.01
    observed_S, observed_I, observed_R = simulator(fiducial_beta, fiducial_gamma, ndays, N)
    observed_data = np.array([observed_S, observed_I, observed_R], ndmin=2)

    # Initialize the ABC SIR object.
    # To use fixed epsilon, set epsilon_fixed to a value.
    # For 75th percentile rule, leave epsilon_fixed as None.
    abc = ABCSIR(observed=observed_data, exponent_scale=0.1, beta_a=1e-2, beta_b=1,
                    simulator_func=simulator, distance_func=distance,
                    ndays=ndays, N=N, n_iterations=100, epsilon_fixed=0.001)
    abc.run()

    # Save results (the .npz file contains S, R, I, epsilon arrays)
    # abc.save_results('/home/ubuntu/iti/data/abc_sir_results.npz')

    # Plotting
    plot_histograms(abc.accepted_betas, abc.accepted_gammas, fiducial_beta, fiducial_gamma)
    if abc.epsilon_fixed is None and len(abc.epsilon_history) > 0:
        plot_epsilon_update(abc.epsilon_history)
    plot_trace(abc.accepted_betas, abc.accepted_gammas, fiducial_beta, fiducial_gamma)