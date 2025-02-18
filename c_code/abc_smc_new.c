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
void summary(double *data, int n, double *median) {
    double *tmp = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        tmp[i] = data[i];
    gsl_sort(tmp, 1, n);
    *median = gsl_stats_median_from_sorted_data(tmp, 1, n);
    // double p90 = gsl_stats_quantile_from_sorted_data(tmp, 1, n, 0.875);
    // double p10 = gsl_stats_quantile_from_sorted_data(tmp, 1, n, 0.125);
    // *spread = p90 - p10;
    free(tmp);
}

// Compute percentile of data using GSL.
double compute_percentile(double *data, int n, double perc) {
    double *tmp = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        tmp[i] = data[i];
    gsl_sort(tmp, 1, n);
    double p = gsl_stats_quantile_from_sorted_data(tmp, 1, n, perc);
    free(tmp);
    return p;
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

void run_abc_smc(const double *obs_S, const double *obs_I, const double *obs_R,
                 int ndays, int N, int I0, int n_generations, int population_size,
                 double exponent_scale, double beta_a, double beta_b,
                 double **final_betas_out, double **final_gammas_out, 
                 double **epsilon_history_out) {
    
    clock_t start_t = clock();
    // Allocate arrays to store particles (each particle: beta and gamma) and distances.
    for (int gen = 0; gen <= n_generations; gen++) {
        double *accepted_beta = malloc(population_size * sizeof(double));
        double *accepted_gamma = malloc(population_size * sizeof(double));
        if (gen == 0) {
            // Initialize particles in the first generation.
            printf("Running Generation 0 ...\n");

            // double *accepted_beta = malloc(population_size * sizeof(double));
            // double *accepted_gamma = malloc(population_size * sizeof(double));
            // double *distances = malloc(population_size * sizeof(double));    
            double (*epsilon_history)[3] = malloc((n_generations) * sizeof(double[3]));

            double d_S, d_I, d_R;

            for (int i = 0; i < population_size; i++) {
                
                double beta = sample_exponential(exponent_scale);
                double gamma = sample_beta(beta_a, beta_b);

                double *sim_S = malloc(ndays * sizeof(int));
                double *sim_I = malloc(ndays * sizeof(int));
                double *sim_R = malloc(ndays * sizeof(int));
                simulator(beta, gamma, ndays, N, I0, sim_S, sim_I, sim_R);

                distance(N, ndays, obs_S, obs_I, obs_R, sim_S, sim_I, sim_R, 
                                &d_S, &d_I, &d_R);

                free(sim_S);
                free(sim_I);
                free(sim_R);

                accepted_beta[i] = beta;
                accepted_gamma[i] = gamma;
            }
            double epsilon_S = compute_percentile(&d_S, population_size, 0.75);
            double epsilon_I = compute_percentile(&d_I, population_size, 0.75);
            double epsilon_R = compute_percentile(&d_R, population_size, 0.75);

            epsilon_history[0][0] = epsilon_S;
            epsilon_history[0][1] = epsilon_I;
            epsilon_history[0][2] = epsilon_R;

            save_csv("abc_smc_gen_0.csv", accepted_beta, accepted_gamma, epsilon_history[0], population_size);
            printf("Generation 0: epsilon_S = %.5f, epsilon_I = %.5f, epsilon_R = %.5f\n",
                epsilon_S, epsilon_I, epsilon_R);
        }
        else{

            printf("Running Generation %d ...\n", gen);
            double *new_particles_beta = malloc(population_size * sizeof(double));
            double *new_particles_gamma = malloc(population_size * sizeof(double));
            // double *new_distances = malloc(population_size * sizeof(double));
            double (*epsilon_history)[3] = malloc((n_generations) * sizeof(double[3]));

            double d_S, d_I, d_R;

            for (int i = 0; i < population_size; i++) {
                int idx = gsl_rng_uniform_int(r, population_size);
                double beta_old = accepted_beta[idx];
                double gamma_old = accepted_gamma[idx];
                double new_beta = beta_old + gsl_ran_gaussian(r, 0.01);
                double new_gamma = gamma_old + gsl_ran_gaussian(r, 0.01);
                if (new_beta <= 0 || new_gamma <= 0 || new_gamma > 1)
                    continue;
                double *sim_S = malloc(ndays * sizeof(int));
                double *sim_I = malloc(ndays * sizeof(int));
                double *sim_R = malloc(ndays * sizeof(int));
                simulator(new_beta, new_gamma, ndays, N, I0, sim_S, sim_I, sim_R);
                distance(N, ndays, obs_S, obs_I, obs_R, sim_S, sim_I, sim_R, 
                                &d_S, &d_I, &d_R);
                free(sim_S);
                free(sim_I);
                free(sim_R);
                if (d_S <= epsilon_history[gen-1][0] && 
                    d_I <= epsilon_history[gen-1][1] && 
                    d_R <= epsilon_history[gen-1][2]) {
                    new_particles_beta[i] = new_beta;
                    new_particles_gamma[i] = new_gamma;
                    // new_distances[i] = d;
                }
            }
        }
    }
}