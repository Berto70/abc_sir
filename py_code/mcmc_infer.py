import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from multiprocessing import Pool
import scipy.stats as s

class SIRMCMC:
    def __init__(self, N, ndays, S0, I0, R0, dI, dR, exponent_scale=1, beta_a=1, beta_b=1):
        self.N = N
        self.ndays = ndays
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.dI = dI
        self.dR = dR
        self.exponent_scale = exponent_scale
        self.beta_a = beta_a
        self.beta_b = beta_b
        
    def prior_func(self):
        """
        Sample parameters from the prior.
        beta ~ Exponential(scale) and gamma ~ Beta(beta_a, beta_b)
        """
        beta = s.expon.rvs(scale=self.exponent_scale)
        gamma = s.beta.rvs(a=self.beta_a, b=self.beta_b)
        return beta, gamma
    
    def log_prior(self, beta, gamma):
        """
        Log prior probabilities for beta and gamma
        """
        log_beta = s.expon.logpdf(beta, scale=self.exponent_scale)
        log_gamma = s.beta.logpdf(gamma, a=self.beta_a, b=self.beta_b)
        
        return log_beta + log_gamma
    
    def log_likelihood(self, beta, gamma):
        """
        Log likelihood based on Poisson distributions for dI and dR
        """
        log_like = 0
        S = self.S0
        I = self.I0
        
        # Sum log probabilities for all timesteps
        for t in range(1, len(self.dI)):
            # Infection likelihood
            lambda_i = beta * S * I / self.N
            log_like += ss.poisson.logpmf(self.dI[t], lambda_i)
            
            # Recovery likelihood
            lambda_r = gamma * I
            log_like += ss.poisson.logpmf(self.dR[t], lambda_r)
            
            # Update states for next iteration
            S = S - self.dI[t]
            I = I + self.dI[t] - self.dR[t]
        
        return log_like
    
    def log_posterior(self, beta, gamma):
        """
        Log posterior combining prior and likelihood
        """
        log_prior = self.log_prior(beta, gamma)
        if np.isinf(log_prior):
            return -np.inf
        return log_prior + self.log_likelihood(beta, gamma)
    
    def run_chain(self, n_samples, init_params, proposal_width=0.01):
        """
        Run MCMC chain with Metropolis-Hastings
        """
        samples = np.zeros((n_samples, 2))
        current = init_params
        accepted = 0
        
        # Current log posterior
        current_log_post = self.log_posterior(*current)
        
        for i in range(n_samples):
            # Propose new parameters
            proposal = current + np.random.normal(0, proposal_width, 2)
            
            # Calculate log posterior for proposal
            proposal_log_post = self.log_posterior(*proposal)
            
            # Accept/reject step
            log_ratio = proposal_log_post - current_log_post
            if log_ratio > np.log(np.random.random()):
                current = proposal
                current_log_post = proposal_log_post
                accepted += 1
                
            samples[i] = current
            
        acceptance_rate = accepted / n_samples
        return samples, acceptance_rate

# Function to run a single chain (for Pool.map)
def run_single_chain(args):
    """
    Run a single MCMC chain
    Args:
        args: tuple containing (mcmc, init_params, n_samples, proposal_width)
    """
    mcmc, init_params, n_samples, proposal_width = args
    chain, acceptance_rate = mcmc.run_chain(n_samples, init_params, proposal_width)
    return chain, acceptance_rate
    
def run_parallel_chains(mcmc, n_chains=3, n_samples=100000, proposal_width=0.01):
    """
    Run multiple MCMC chains in parallel using multiprocessing
    """
    # Define different starting points spread out in parameter space
    init_points = [
        np.array([0.15, 0.015]),  # Higher than fiducial
        np.array([0.05, 0.005]),  # Lower than fiducial
        np.array([0.2, 0.02])     # Much higher than fiducial
    ]
    
    # Create argument tuples for each chain
    args = [(mcmc, init_point, n_samples, proposal_width) 
            for init_point in init_points[:n_chains]]
    
    # Run chains in parallel
    with Pool() as pool:
        results = pool.map(run_single_chain, args)
    
    # Unpack results
    chains = [r[0] for r in results]
    acceptance_rates = [r[1] for r in results]
    
    return chains, acceptance_rates

def gelman_rubin_diagnostic(chains, burn_in=0):
    """
    Calculate Gelman-Rubin diagnostic (R̂) for each parameter
    R̂ close to 1 indicates convergence (typically < 1.1 is considered acceptable)
    """
    # Remove burn-in and reshape chains
    n_chains = len(chains)
    n_samples = chains[0].shape[0] - burn_in
    n_params = chains[0].shape[1]
    
    # Reshape to (n_params, n_chains, n_samples)
    chain_data = np.array([chain[burn_in:] for chain in chains])
    chain_data = np.transpose(chain_data, (2, 0, 1))  # shape: (n_params, n_chains, n_samples)
    
    R_hats = []
    for param_chains in chain_data:  # For each parameter
        # Calculate within-chain and between-chain variances
        chain_means = np.mean(param_chains, axis=1)
        chain_vars = np.var(param_chains, axis=1, ddof=1)
        
        # Within-chain variance (mean of variances)
        W = np.mean(chain_vars)
        
        # Between-chain variance
        B = n_samples * np.var(chain_means, ddof=1)
        
        # Calculate pooled variance estimate
        var_plus = ((n_samples - 1) * W + B) / n_samples
        
        # Calculate R̂
        R_hat = np.sqrt(var_plus / W)
        R_hats.append(R_hat)
    
    return np.array(R_hats)

def save_chain_data(chain, filename):
    """
    Save chain data to CSV file
    """
    df = pd.DataFrame(chain, columns=['beta', 'gamma'])
    df.to_csv(filename, index=False)
    print(f"Chain data saved to {filename}")

def load_chain_data(filename):
    """
    Load chain data from CSV file
    """
    return pd.read_csv(filename).values

def plot_multiple_chains(chains, fiducial_values, burn_in=0, param_names=['β', 'γ']):
    """
    Plot diagnostic plots for multiple MCMC chains
    """
    n_params = chains[0].shape[1]
    n_chains = len(chains)
    colors = plt.cm.viridis(np.linspace(0, 1, n_chains))
    
    # Trace plots
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 8))
    for i in range(n_params):
        for j, chain in enumerate(chains):
            axes[i].plot(chain[:, i], color=colors[j], alpha=0.5, 
                        label=f'Chain {j+1}')
        axes[i].axhline(y=fiducial_values[i], color='r', linestyle='--',
                       label=f'Fiducial value: {fiducial_values[i]}')
        axes[i].axvline(x=burn_in, color='g', linestyle='--',
                       label=f'Burn-in: {burn_in}')
        axes[i].set_title(f'Trace Plot - {param_names[i]}')
        axes[i].legend()
    plt.tight_layout()
    plt.show()
    
    # Combined histograms
    fig, axes = plt.subplots(1, n_params, figsize=(15, 5))
    for i in range(n_params):
        for j, chain in enumerate(chains):
            sns.histplot(chain[burn_in:, i], ax=axes[i], alpha=0.5,
                        label=f'Chain {j+1}', color=colors[j])
        axes[i].axvline(x=fiducial_values[i], color='r', linestyle='--',
                       label=f'Fiducial value: {fiducial_values[i]}')
        axes[i].set_title(f'Histogram - {param_names[i]}')
        axes[i].legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Simulation parameters
    N = 1000  # population size
    ndays = 600  # simulation days
    fiducial_beta = 0.1
    fiducial_gamma = 0.01
    
    # Generate synthetic data
    def generate_sir_data(beta, gamma, ndays, N, I0=10):
        """
        Generate synthetic SIR data for testing
        """
        S = np.zeros(ndays)
        I = np.zeros(ndays)
        R = np.zeros(ndays)
        dI = np.zeros(ndays)
        dR = np.zeros(ndays)
        frac_infected = np.zeros(ndays)
        
        S[0] = N - I0
        I[0] = I0
        R[0] = 0
        
        frac_infected[0] = I0 / N
        for t in range(1, ndays):
            # new infections and recoveries
            dI[t] = np.random.binomial(int(S[t-1]), 1 - np.exp(-beta * frac_infected[t-1]))
            dR[t] = np.random.binomial(int(I[t-1]), gamma)
            
            # Update states
            S[t] = S[t-1] - dI[t]
            I[t] = I[t-1] + dI[t] - dR[t]
            R[t] = R[t-1] + dR[t]
            
            # Ensure non-negative values
            S[t] = max(0, S[t])
            I[t] = max(0, I[t])
            R[t] = max(0, R[t])
        
        return S, I, R, dI, dR

    S, I, R, dI, dR = generate_sir_data(fiducial_beta, fiducial_gamma, ndays, N, I0=10)
    
    # Create MCMC object
    mcmc = SIRMCMC(N, ndays, S[0], I[0], R[0], dI, dR)
    
    # Run parallel chains
    print("Running parallel chains...")
    chains, acceptance_rates = run_parallel_chains(mcmc)
    
    # Print acceptance rates
    for i, rate in enumerate(acceptance_rates):
        print(f"Chain {i+1} acceptance rate: {rate:.2%}")
    
    # Calculate and print Gelman-Rubin statistics
    burn_in = 500  # Adjust based on trace plots
    r_hats = gelman_rubin_diagnostic(chains, burn_in)
    for i, param in enumerate(['beta', 'gamma']):
        print(f"\nGelman-Rubin R̂ for {param}: {r_hats[i]:.4f}")
    
    # Plot diagnostics for all chains
    plot_multiple_chains(chains, [fiducial_beta, fiducial_gamma], burn_in)
    
    # Save all chain data
    for i, chain in enumerate(chains):
        save_chain_data(chain, f'mcmc_chain_{i+1}_bin.csv')
    
    # Calculate summary statistics for all chains combined
    all_chains = np.vstack(chains)
    for i, param in enumerate(['beta', 'gamma']):
        median = np.median(all_chains[burn_in:, i])
        mean = np.mean(all_chains[burn_in:, i])
        ci_95 = np.percentile(all_chains[burn_in:, i], [2.5, 97.5])
