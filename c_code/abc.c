#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_sort.h>
#include <omp.h>

// Global GSL random number generator.
gsl_rng *r;

// Binomial sampler using GSL.
int binomial(int n, double p) {
    return (int)gsl_ran_binomial(r, p, n);
}

// Sample from an exponential distribution using GSL.
double sample_exponential(double scale) {
    return gsl_ran_exponential(r, scale);
}

// Box-Muller transform for standard normal sample replaced by GSL Gaussian.
double sample_normal() {
    return gsl_ran_gaussian(r, 1.0);
}

// Sample from Gamma distribution using GSL.
double sample_gamma(double shape, double scale) {
    return gsl_ran_gamma(r, shape, scale);
}

// Sample from Beta distribution using GSL.
double sample_beta(double a, double b) {
    return gsl_ran_beta(r, a, b);
}

// Simulator: run an Euler integration SIR model over ndays.
// S, I, and R arrays must be preallocated of length ndays.
void simulator(double beta, double gamma, int ndays, int N, int I0, int *S, int *I, int *R) {
    double *frac_infected = (double *)malloc(ndays * sizeof(double));
    int *dI = (int *)malloc(ndays * sizeof(int));
    int *dR = (int *)malloc(ndays * sizeof(int));

    S[0] = N - I0;
    I[0] = I0;
    R[0] = 0;
    frac_infected[0] = (double) I0 / N;

    for (int t = 1; t < ndays; t++) {
        dI[t] = binomial(S[t-1], 1 - exp(-beta * frac_infected[t-1]));
        dR[t] = binomial(I[t-1], gamma);
        S[t] = S[t-1] - dI[t];
        I[t] = I[t-1] + dI[t] - dR[t];
        R[t] = R[t-1] + dR[t];
        frac_infected[t] = (double) I[t] / N;
        if (S[t] + I[t] + R[t] != N) {
            fprintf(stderr, "Error: S+I+R != N at time %d\n", t);
            exit(EXIT_FAILURE);
        }
    }
    free(frac_infected);
    free(dI);
    free(dR);
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

// Compute distance between observed and simulated data (S,I,R arrays).
// d_S, d_I, d_R are normalized by population N.
void distance(int N, int ndays, int *obs_S, int *obs_I, int *obs_R, 
              int *sim_S, int *sim_I, int *sim_R,
              double *d_S, double *d_I, double *d_R) {
    *d_S = compute_norm(sim_S, obs_S, ndays) / N;
    *d_I = compute_norm(sim_I, obs_I, ndays) / N;
    *d_R = compute_norm(sim_R, obs_R, ndays) / N;
}

// Summary statistic using GSL: compute median and spread (90th percentile minus 10th percentile).
void summary(double *data, int n, double *median, double *spread) {
    double *tmp = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        tmp[i] = data[i];
    gsl_sort(tmp, 1, n);
    *median = gsl_stats_median_from_sorted_data(tmp, 1, n);
    double p90 = gsl_stats_quantile_from_sorted_data(tmp, 1, n, 0.875);
    double p10 = gsl_stats_quantile_from_sorted_data(tmp, 1, n, 0.125);
    *spread = p90 - p10;
    free(tmp);
}

// Add the following function before main():
void save_csv(const char *filename, double *betas, double *gammas, int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open file %s for writing.\n", filename);
        return;
    }
    fprintf(fp, "beta,gamma\n");
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%.6f,%.6f\n", betas[i], gammas[i]);
    }
    fclose(fp);
}

void save_iter(const char *filename, double *eps, double *ttime, double *acc_rate, double *avg_trial, int n) {
    // Open the file in append mode to preserve existing content.
    FILE *fp = fopen(filename, "a");
    if (!fp) {
        fprintf(stderr, "Error: could not open file %s for appending.\n", filename);
        return;
    }
    
    // If the file is empty, write the header.
    fseek(fp, 0, SEEK_END);
    if (ftell(fp) == 0) {
        fprintf(fp, "eps,time,acc_rate,avg_trial\n");
    }
    
    // Append new data.
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%02.0f,%.6f,%.6f,%.6f\n", eps[i], ttime[i], acc_rate[i], avg_trial[i]);
    }
    fclose(fp);
}


/*int main() {
    // Initialize the GSL random number generator.
    const gsl_rng_type * T;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);

    clock_t start_t = clock();

    int ndays = 600, N = 1000, I0 = 10;
    double fiducial_beta = 0.1, fiducial_gamma = 0.01;
    int n_iterations = 10000;      // number of accepted samples to collect
    double epsilon_fixed = 1;  // acceptance threshold

    // Allocate observed data arrays.
    int *obs_S = (int *)malloc(ndays * sizeof(int));
    int *obs_I = (int *)malloc(ndays * sizeof(int));
    int *obs_R = (int *)malloc(ndays * sizeof(int));

    // Create a mock dataset using fiducial parameters.
    simulator(fiducial_beta, fiducial_gamma, ndays, N, I0, obs_S, obs_I, obs_R);

    // Prior parameters.
    double exponent_scale = 0.1, beta_a = 0.01, beta_b = 1.0;
    // Allocate arrays for accepted samples.
    double *accepted_betas = (double *)malloc(n_iterations * sizeof(double));
    double *accepted_gammas = (double *)malloc(n_iterations * sizeof(double));
    int total_proposals = 0;
    int total_trials = 0;

    // For each accepted sample.
    for (int iter = 0; iter < n_iterations; iter++) {
        int trials = 0;
        while (1) {
            trials++;
            total_proposals++;
            // Sample from prior using GSL functions.
            double beta = sample_exponential(exponent_scale);
            double gamma = sample_beta(beta_a, beta_b);

            // Allocate arrays for simulation output.
            int *sim_S = (int *)malloc(ndays * sizeof(int));
            int *sim_I = (int *)malloc(ndays * sizeof(int));
            int *sim_R = (int *)malloc(ndays * sizeof(int));

            simulator(beta, gamma, ndays, N, I0, sim_S, sim_I, sim_R);

            double d_S, d_I, d_R;
            distance(N, ndays, obs_S, obs_I, obs_R, sim_S, sim_I, sim_R, &d_S, &d_I, &d_R);

            // if (iter % 10 == 0) {
            //     printf("Iteration %d: d_S = %.3f, d_I = %.3f, d_R = %.3f\n", iter, d_S, d_I, d_R);
            // }
            // Free simulation arrays.
            free(sim_S);
            free(sim_I);
            free(sim_R);

            // Check acceptance.
            if (d_S <= epsilon_fixed && d_I <= epsilon_fixed && d_R <= epsilon_fixed) {
                accepted_betas[iter] = beta;
                accepted_gammas[iter] = gamma;
                total_trials += trials;
                break;
            }
        }
    }
    double acceptance_rate = (double)n_iterations / total_proposals;
    printf("Acceptance rate: %.3f, Average number of trials: %.3f\n",
           acceptance_rate, (double)total_trials / n_iterations);

    // Compute medians on accepted parameters.
    double median_beta, median_gamma, spread_dummy;
    summary(accepted_betas, n_iterations, &median_beta, &spread_dummy);
    summary(accepted_gammas, n_iterations, &median_gamma, &spread_dummy);
    printf("Median beta: %.3f\n", median_beta);
    printf("Median gamma: %.3f\n", median_gamma);

    clock_t end_t = clock();
    double exec_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("Execution time: %.3f seconds\n", exec_t);

    // Near the end of main() before freeing memory:
    save_csv("/home/ubuntu/iti/project/C/abc_data/abc_10k_01.csv", 
            accepted_betas, accepted_gammas, n_iterations);

    free(obs_S);
    free(obs_I);
    free(obs_R);
    free(accepted_betas);
    free(accepted_gammas);

    // Free the GSL random number generator.
    gsl_rng_free(r);

    return 0;
}*/

int main() {
    // Initialize the GSL random number generator.
    const gsl_rng_type * T;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);

    clock_t start_t = clock();

    int ndays = 600, N = 1000, I0 = 10;
    double fiducial_beta = 0.2, fiducial_gamma = 0.03;
    int n_iterations = 10000;      // number of accepted samples to collect
    
    int eps_list[] = {20, 18, 16, 14, 10, 6, 5, 3, 2, 1, 0.5, 0.1};;
    int eps_list_length = sizeof(eps_list) / sizeof(eps_list[0]);
    
    // #pragma omp parallel for num_threads(2)
    for (int i = 0; i < eps_list_length; i++) {

        double *exec_t = (double *)malloc(1 * sizeof(double));
        double *acceptance_rate = (double *)malloc(1 * sizeof(double));
        double *avg_trial = (double *)malloc(1 * sizeof(double));

        // Allocate observed data arrays.
        int *obs_S = (int *)malloc(ndays * sizeof(int));
        int *obs_I = (int *)malloc(ndays * sizeof(int));
        int *obs_R = (int *)malloc(ndays * sizeof(int));

        // Create a mock dataset using fiducial parameters.
        simulator(fiducial_beta, fiducial_gamma, ndays, N, I0, obs_S, obs_I, obs_R);

        // Prior parameters.
        double exponent_scale = 0.2, beta_a = 0.03, beta_b = 1.0;
        // Allocate arrays for accepted samples.
        double *accepted_betas = (double *)malloc(n_iterations * sizeof(double));
        double *accepted_gammas = (double *)malloc(n_iterations * sizeof(double));

        double epsilon_fixed = eps_list[i];  // acceptance threshold

        int total_proposals = 0;
        int total_trials = 0;

        // For each accepted sample.
        for (int iter = 0; iter < n_iterations; iter++) {
            int trials = 0;
            while (1) {
                trials++;
                total_proposals++; 
                // Sample from prior using GSL functions.
                double beta = sample_exponential(exponent_scale);
                double gamma = sample_beta(beta_a, beta_b);

                // Allocate arrays for simulation output.
                int *sim_S = (int *)malloc(ndays * sizeof(int));
                int *sim_I = (int *)malloc(ndays * sizeof(int));
                int *sim_R = (int *)malloc(ndays * sizeof(int));

                simulator(beta, gamma, ndays, N, I0, sim_S, sim_I, sim_R);

                double d_S, d_I, d_R;
                distance(N, ndays, obs_S, obs_I, obs_R, sim_S, sim_I, sim_R, 
                        &d_S, &d_I, &d_R);

                // if (iter % 10 == 0) {
                //     printf("Iteration %d: d_S = %.3f, d_I = %.3f, d_R = %.3f\n", iter, d_S, d_I, d_R);
                // }
                // Free simulation arrays.
                free(sim_S);
                free(sim_I);
                free(sim_R);

                // Check acceptance.
                if (d_S <= epsilon_fixed && 
                    d_I <= epsilon_fixed && 
                    d_R <= epsilon_fixed) {

                    accepted_betas[iter] = beta;
                    accepted_gammas[iter] = gamma;
                    total_trials += trials;
                    break; 
                }
            }
        }
        *acceptance_rate = (double)n_iterations / total_proposals;
        *avg_trial = (double)total_trials / n_iterations;
        printf("Acceptance rate: %.3f, Average number of trials: %.3f\n",
            *acceptance_rate, *avg_trial);

        // Compute medians on accepted parameters.
        double median_beta, median_gamma, spread_dummy;
        summary(accepted_betas, n_iterations, &median_beta, &spread_dummy);
        summary(accepted_gammas, n_iterations, &median_gamma, &spread_dummy);
        printf("Median beta: %.3f\n", median_beta);
        printf("Median gamma: %.3f\n", median_gamma);

        clock_t end_t = clock();
        *exec_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
        printf("Execution time: %.3f seconds\n", *exec_t);

        save_iter("/home/ubuntu/iti/project/C/abc_data/abc_10k_02003_iter.csv", 
                &epsilon_fixed, exec_t, acceptance_rate, avg_trial, 1);

        char filename[256];
        sprintf(filename, "/home/ubuntu/iti/project/C/abc_data/abc_10k_02003_%02d.csv", 
                (int)epsilon_fixed);
        // Near the end of main() before freeing memory:
        save_csv(filename, 
                accepted_betas, accepted_gammas, n_iterations);

        free(obs_S);
        free(obs_I);
        free(obs_R);
        free(accepted_betas);
        free(accepted_gammas);
        free(exec_t);
        free(acceptance_rate);
        free(avg_trial);
    }

    // Free the GSL random number generator.
    gsl_rng_free(r);

    return 0;
}