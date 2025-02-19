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
            free(frac_infected);
            free(dI);
            free(dR);
            exit(EXIT_FAILURE);
        }
    }
    free(frac_infected);
    free(dI);
    free(dR);
}

// Compute Euclidean norm difference between two int arrays.
double compute_norm(const int *a, const int *b, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Compute distance between observed and simulated data (S,I,R arrays).
// d_S, d_I, d_R are normalized by population N.
void distance(int N, int ndays, const int *obs_S, const int *obs_I, const int *obs_R, 
              int *sim_S, int *sim_I, int *sim_R,
              double *d_S, double *d_I, double *d_R) {
    *d_S = compute_norm((int*)obs_S, sim_S, ndays) / N;
    *d_I = compute_norm((int*)obs_I, sim_I, ndays) / N;
    *d_R = compute_norm((int*)obs_R, sim_R, ndays) / N;
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

void weight(int population_size, double beta_old, double beta_new, 
            double gamma_old, double gamma_new, 
            double exponent_scale, double beta_a, double beta_b,
            double w_beta_old, double w_gamma_old,
            double *w_beta, double *w_gamma) {
    double sum_beta = 0;
    double sum_gamma = 0;
    for (int j = 0; j < population_size; j++) {
        sum_beta += w_beta_old * gsl_ran_gaussian_pdf(beta_old, 0.01);
    }
    *w_beta = (sample_exponential(exponent_scale)) / sum_beta;

    for (int j = 0; j < population_size; j++) {
        sum_gamma += w_gamma_old * gsl_ran_gaussian_pdf(gamma_old, 0.01);
    }
    *w_gamma = (sample_beta(beta_a, beta_b)) / sum_gamma;
}

int weighted_sample(gsl_rng *rng, const double *weights, int npart) {
    double r = gsl_rng_uniform(rng);  // Random number between 0 and 1
    double cum_sum = 0.0;
    
    for (int i = 0; i < npart; i++) {
        cum_sum += weights[i];
        if (r <= cum_sum) {
            return i;
        }
    }
    
    return npart - 1;  // Fallback to last element
}

void run_abc_smc(const int *obs_S, const int *obs_I, const int *obs_R,
                 int ndays, int N, int I0, int n_generations, int population_size,
                 double exponent_scale, double beta_a, double beta_b) {
    
    // Allocate control arrays.
    double *acceptance_rate = (double *)malloc(1 * sizeof(double));
    double *avg_trial = (double *)malloc(1 * sizeof(double));

    // Allocate samples arrays.
    double *particles_beta = malloc(population_size * sizeof(double));
    double *particles_gamma = malloc(population_size * sizeof(double));
    // double *distances = malloc(population_size * sizeof(double));
    double (*epsilon_history)[3] = malloc((n_generations) * sizeof(double[3]));

    double (*weight_beta)[n_generations] = malloc(population_size * sizeof(double[n_generations]));
    double (*weight_gamma)[n_generations] = malloc(population_size * sizeof(double[n_generations])); 

    for (int gen = 0; gen <= n_generations; gen++) {
        // Generation 0: sample from prior.
        if (gen == 0) {
            printf("Running Generation 0 ...\n");
            for (int i = 0; i < population_size; i++) {
                double beta = sample_exponential(exponent_scale);
                double gamma = sample_beta(beta_a, beta_b);
                int *sim_S = malloc(ndays * sizeof(int));
                int *sim_I = malloc(ndays * sizeof(int));
                int *sim_R = malloc(ndays * sizeof(int));
                simulator(beta, gamma, ndays, N, I0, sim_S, sim_I, sim_R);

                double d_S, d_I, d_R;
                distance(N, ndays, obs_S, obs_I, obs_R, sim_S, sim_I, sim_R,
                        &d_S, &d_I, &d_R); 

                epsilon_history[0][0] = compute_percentile(&d_S, population_size, 0.75);
                epsilon_history[0][1] = compute_percentile(&d_I, population_size, 0.75);
                epsilon_history[0][2] = compute_percentile(&d_R, population_size, 0.75);

                particles_beta[i] = beta;
                particles_gamma[i] = gamma;

                weight_beta[0][i] = 1.;
                weight_gamma[0][i] = 1.;

                free(sim_S); free(sim_I); free(sim_R);
            }
            save_csv("/home/ubuntu/abc_sir/c_code/abc_smc_data/particles_gen0.csv", 
            particles_beta, particles_gamma, epsilon_history[0], population_size);
        } else {
                for (int i = 0; i < population_size; i++) {

                    int idx_beta = weighted_sample(r, weight_beta[gen-1], population_size);
                    int idx_gamma = weighted_sample(r, weight_gamma[gen-1], population_size);
                    double old_beta = particles_beta[idx_beta];
                    double old_gamma = particles_gamma[idx_gamma];

                    double new_beta = old_beta + gsl_ran_gaussian(r, 0.01);
                    double new_gamma = old_gamma + gsl_ran_gaussian(r, 0.01);

                    int *sim_S = malloc(ndays * sizeof(int));
                    int *sim_I = malloc(ndays * sizeof(int));
                    int *sim_R = malloc(ndays * sizeof(int));
                    simulator(new_beta, new_gamma, ndays, N, I0, sim_S, sim_I, sim_R);

                    double d_S, d_I, d_R;
                    distance(N, ndays, obs_S, obs_I, obs_R, sim_S, sim_I, sim_R,
                            &d_S, &d_I, &d_R);
                    epsilon_history[gen][0] = compute_percentile(&d_S, population_size, 0.75);
                    epsilon_history[gen][1] = compute_percentile(&d_I, population_size, 0.75);
                    epsilon_history[gen][2] = compute_percentile(&d_R, population_size, 0.75);

                    if (d_S < epsilon_history[gen-1][0] &&
                        d_I < epsilon_history[gen-1][1] &&
                        d_R < epsilon_history[gen-1][2]) {
                        particles_beta[i] = new_beta;
                        particles_gamma[i] = new_gamma;
                        weight(population_size, old_beta, 
                                    new_beta, old_gamma, new_gamma, exponent_scale, 
                                    beta_a, beta_b, weight_beta[gen-1][idx_beta], 
                                    weight_gamma[gen-1][idx_gamma], &weight_beta[gen][i], &weight_gamma[gen][i]);
                        
                    }
                }
                {
            char filename[256];
            sprintf(filename, "/home/ubuntu/abc_sir/c_code/abc_data/particles_gen_%d.csv", gen);
            FILE *fp = fopen(filename, "w");
            if (fp) {
                fprintf(fp, "beta,gamma,epsilon\n");
                for (int i = 0; i < population_size; i++) {
                    fprintf(fp, "%.6f,%.6f,%.6f\n", particles_beta[i], particles_gamma[i], epsilon_history[gen][0]);
                }
                fclose(fp);
            }
        }
            }
    }
}

int main () {

    // Initialize the GSL random number generator.
    const gsl_rng_type * T;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);

    double *exec_t = (double *)malloc(1 * sizeof(double));

    clock_t start_t = clock();

    int ndays = 600, N = 1000, I0 = 10;
    double fiducial_beta = 0.1, fiducial_gamma = 0.01;
    int population_size = 10000;      // number of accepted samples to collect
    int n_generations = 3;
    
    // Prior parameters.
    double exponent_scale = 0.1, beta_a = 0.01, beta_b = 1.0;

    // Allocate observed data arrays.
    int *obs_S = (int *)malloc(ndays * sizeof(int));
    int *obs_I = (int *)malloc(ndays * sizeof(int));
    int *obs_R = (int *)malloc(ndays * sizeof(int));

    // Create a mock dataset using fiducial parameters.
    simulator(fiducial_beta, fiducial_gamma, ndays, N, I0, obs_S, obs_I, obs_R);

    run_abc_smc(obs_S, obs_I, obs_R, ndays, N, I0, n_generations, population_size, 
                exponent_scale, beta_a, beta_b);

    clock_t end_t = clock();
    *exec_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("Execution time: %.3f seconds\n", *exec_t);
}