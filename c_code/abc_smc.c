#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_sort.h>

#define DEFAULT_I0 10

// Helper: Compute percentile from an array (copy it and use gsl_sort)
double compute_percentile(const double *data, int n, double perc) {
    double *copy = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        copy[i] = data[i];
    gsl_sort(copy, 1, n);
    double pos = (n - 1) * perc / 100.0;
    int idx = (int)floor(pos);
    double frac = pos - idx;
    double perc_val = copy[idx] + frac * ((idx+1 < n ? copy[idx+1] : copy[idx]) - copy[idx]);
    free(copy);
    return perc_val;
}

// Simulator: computes S, I, R arrays over ndays given beta and gamma.
void simulator(gsl_rng *rng, double beta, double gamma, int ndays, int N, 
                int I0, double *S, double *I, double *R) {
    // ...allocate temporary arrays for frac_infected, dI, dR...
    double *frac_infected = calloc(ndays, sizeof(double));
    double *dI = calloc(ndays, sizeof(double));
    double *dR = calloc(ndays, sizeof(double));
    
    S[0] = N - I0;
    I[0] = I0;
    R[0] = 0;
    frac_infected[0] = (double)I0 / N;
    
    for (int t = 1; t < ndays; t++) {
        unsigned int nS = (unsigned int)S[t-1];
        double pI = 1 - exp(-beta * frac_infected[t-1]);
        // Binomial draw for new infections
        dI[t] = gsl_ran_binomial(rng, pI, nS);
        
        unsigned int nI = (unsigned int)I[t-1];
        dR[t] = gsl_ran_binomial(rng, gamma, nI);
        
        S[t] = S[t-1] - dI[t];
        I[t] = I[t-1] + dI[t] - dR[t];
        R[t] = R[t-1] + dR[t];
        
        frac_infected[t] = I[t] / (double)N;
        
        if ((int)(S[t] + I[t] + R[t]) != N) {
            fprintf(stderr, "Error at day %d: S+I+R != N\n", t);
            exit(EXIT_FAILURE);
        }
    }
    free(frac_infected);
    free(dI);
    free(dR);
}

// Summary: compute mean and spread (90th - 10th percentile) using GSL functions.
void summary(const double *data, size_t n, double out[2]) {
    out[0] = gsl_stats_mean(data, 1, n); // mean as "median"
    double p90 = compute_percentile(data, n, 90.0);
    double p10 = compute_percentile(data, n, 10.0);
    out[1] = p90 - p10;
}

// Compute Euclidean norm difference between two int arrays.
double compute_norm(int *a, int *b, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Distance: computes L2 norm differences for S, I, R arrays.
double distance(int N, int ndays, const double *obs_S, const double *obs_I, 
                const double *obs_R, const double *sim_S, const double *sim_I, 
                const double *sim_R) {
    double sumS = 0.0, sumI = 0.0, sumR = 0.0;
    
    sumS = compute_norm(sim_S, obs_S, ndays);
    sumI = compute_norm(sim_I, obs_I, ndays);
    sumR = compute_norm(sim_R, obs_R, ndays);

    double d_S = sqrt(sumS) / N;
    double d_I = sqrt(sumI) / N;
    double d_R = sqrt(sumR) / N;
    return d_S + d_I + d_R;
}

// Sample parameters from prior: beta ~ Exponential(scale), gamma ~ Beta(a, b)
void sample_prior(gsl_rng *rng, double exponent_scale, double beta_a, 
                 double beta_b, double *beta, double *gamma) {
    *beta = gsl_ran_exponential(rng, exponent_scale);
    *gamma = gsl_ran_beta(rng, beta_a, beta_b);
}

// Add the following function before main():
void save_csv(const char *filename, double *betas, double *gammas, 
             double *epsilons, int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open file %s for writing.\n", filename);
        return;
    }
    fprintf(fp, "beta,gamma,epsilon\n");
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%.6f,%.6f,%.6f\n", betas[i], gammas[i], epsilons[i]);
    }
    fclose(fp);
}

// Run ABC-SMC algorithm using plain functions.
void run_abc_smc(gsl_rng *rng, const double *obs_S, const double *obs_I, 
                const double *obs_R, int ndays, int N, int n_generations, 
                int population_size, double exponent_scale, double beta_a, 
                double beta_b, double **final_betas, double **final_gammas, 
                double **epsilon_history_out) {
    // Allocate arrays to store particles (each particle: beta and gamma) and distances.
    double *particles_beta = malloc(population_size * sizeof(double));
    double *particles_gamma = malloc(population_size * sizeof(double));
    double *distances = malloc(population_size * sizeof(double));
    
    double *epsilon_history = malloc((n_generations) * sizeof(double));
    
    // Generation 0: sample from prior.
    printf("Running Generation 0 ...\n");
    clock_t start_t = clock();
    for (int i = 0; i < population_size; i++) {
        double beta, gamma;
        sample_prior(rng, exponent_scale, beta_a, beta_b, &beta, &gamma);
        // Allocate simulator arrays.
        double *S = malloc(ndays * sizeof(double));
        double *I = malloc(ndays * sizeof(double));
        double *R = malloc(ndays * sizeof(double));
        simulator(rng, beta, gamma, ndays, N, DEFAULT_I0, S, I, R);
        distances[i] = distance(N, ndays, obs_S, obs_I, obs_R, S, I, R);
        particles_beta[i] = beta;
        particles_gamma[i] = gamma;
        free(S); free(I); free(R);
    }
    double epsilon = compute_percentile(distances, population_size, 75.0);
    epsilon_history[0] = epsilon;
    printf("Generation 0: epsilon = %.5f\n", epsilon);
    
    // For subsequent generations.
    for (int gen = 1; gen < n_generations; gen++) {
        double *new_particles_beta = malloc(population_size * sizeof(double));
        double *new_particles_gamma = malloc(population_size * sizeof(double));
        double *new_distances = malloc(population_size * sizeof(double));
        int count = 0;
        printf("Running Generation %d ...\n", gen);
        while (count < population_size) {
            int idx = gsl_rng_uniform_int(rng, population_size);
            double beta_old = particles_beta[idx];
            double gamma_old = particles_gamma[idx];
            // Perturbation with Gaussian kernel, std=0.01.
            double new_beta = beta_old + gsl_ran_gaussian(rng, 0.01);
            double new_gamma = gamma_old + gsl_ran_gaussian(rng, 0.01);
            if (new_beta <= 0 || new_gamma <= 0 || new_gamma > 1)
                continue;
            double *S = malloc(ndays * sizeof(double));
            double *I = malloc(ndays * sizeof(double));
            double *R = malloc(ndays * sizeof(double));
            simulator(rng, new_beta, new_gamma, ndays, N, DEFAULT_I0, S, I, R);
            double d = distance(N, ndays, obs_S, obs_I, obs_R, S, I, R);
            free(S); free(I); free(R);
            if (d <= epsilon) {
                new_particles_beta[count] = new_beta;
                new_particles_gamma[count] = new_gamma;
                new_distances[count] = d;
                count++;
            }
            }
        // Update epsilon as the 75th percentile of new distances.
        epsilon = compute_percentile(new_distances, population_size, 75.0);
        epsilon_history[gen] = epsilon;
        printf("Generation %d: epsilon = %.5f\n", gen, epsilon);

        {
            char filename[256];
            sprintf(filename, "/home/ubuntu/iti/project/C/smc_data/abc_smc_gen_%d.csv", gen);
            FILE *fp = fopen(filename, "w");
            if (fp) {
                fprintf(fp, "beta,gamma,epsilon\n");
                for (int i = 0; i < population_size; i++) {
                    fprintf(fp, "%.6f,%.6f,%.6f\n", new_particles_beta[i], 
                            new_particles_gamma[i], new_distances[i]);
                }
                fclose(fp);
            }
        }
        
        // Replace old particles.
        free(particles_beta);
        free(particles_gamma);
        free(distances);
        particles_beta = new_particles_beta;
        particles_gamma = new_particles_gamma;
        distances = new_distances;
    }
    clock_t end_t = clock();
    double time_taken = ((double)(end_t - start_t)) / CLOCKS_PER_SEC;
    printf("Time taken: %.2f seconds\n", time_taken);
    
    *final_betas = particles_beta;   // Caller is responsible for freeing these arrays.
    *final_gammas = particles_gamma;
    *epsilon_history_out = epsilon_history;
    
    free(distances);
}

// ---- Main function ----
int main(void) {
    // Initialize GSL RNG.
    const gsl_rng_type *T;
    gsl_rng *rng;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    rng = gsl_rng_alloc(T);
    
    int ndays = 600;
    int N = 1000;
    // int sim_I0 = 10;
    
    // Create a mock dataset using fiducial parameters.
    double fiducial_beta = 0.1;
    double fiducial_gamma = 0.01;
    double *obs_S = malloc(ndays * sizeof(double));
    double *obs_I = malloc(ndays * sizeof(double));
    double *obs_R = malloc(ndays * sizeof(double));
    simulator(rng, fiducial_beta, fiducial_gamma, ndays, N, 
             DEFAULT_I0, obs_S, obs_I, obs_R);
    
    // Parameters for ABC-SMC.
    int n_generations = 11;
    int population_size = 10000;
    double exponent_scale = 0.1;
    double beta_a = 1e-2;
    double beta_b = 1;
    
    double *final_betas = NULL;
    double *final_gammas = NULL;
    double *epsilon_history = NULL;
    
    run_abc_smc(rng, obs_S, obs_I, obs_R, ndays, N, n_generations, 
                population_size, exponent_scale, beta_a, beta_b,
                &final_betas, &final_gammas, &epsilon_history);
    
    // For simplicity, print the first 5 final beta/gamma samples and epsilon history.
    printf("Final accepted beta samples (first 5):\n");
    for (int i = 0; i < (population_size < 5 ? population_size : 5); i++) {
        printf("%.5f ", final_betas[i]);
    }
    printf("\nFinal accepted gamma samples (first 5):\n");
    for (int i = 0; i < (population_size < 5 ? population_size : 5); i++) {
        printf("%.5f ", final_gammas[i]);
    }
    printf("\nEpsilon history:\n");
    for (int i = 0; i < n_generations; i++) {
        printf("%.5f ", epsilon_history[i]);
    }
    printf("\n");

    // Save the final beta/gamma samples and epsilon history to a CSV file.
    save_csv("/home/ubuntu/iti/project/C/abc_smc_final.csv", final_betas, 
            final_gammas, epsilon_history, population_size);

    // Free allocated memory.
    free(obs_S); free(obs_I); free(obs_R);
    free(final_betas);
    free(final_gammas);
    free(epsilon_history);
    gsl_rng_free(rng);
    
    return 0;
}
