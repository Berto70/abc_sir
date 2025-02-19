#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_sort.h>
// Declarations for functions from other files (assumed prototypes exist)
// void simulator(double beta, double gamma, int ndays, int N, int I0, int *S, int *I, int *R);
// double sample_exponential(double scale);
// double sample_beta(double a, double b);
// void weight(int population_size, double beta_old, double beta_new, 
//             double gamma_old, double gamma_new, 
//             double exponent_scale, double beta_a, double beta_b,
//             double w_beta_old, double w_gamma_old,
//             double *w_beta, double *w_gamma);
// void distance(int N, int ndays,
//               const int *obs_S, const int *obs_I, const int *obs_R, 
//               const int *sim_S, const int *sim_I, const int *sim_R,
//               double *d_S, double *d_I, double *d_R);
// double compute_percentile(double *data, int n, double perc);

// Global GSL random number generator.
gsl_rng *r = NULL;

/* ---------------------------------------------------------------------
   Sampling functions using GSL
   --------------------------------------------------------------------- */
int binomial(int n, double p) {
    return (int)gsl_ran_binomial(r, p, n);
}

double sample_exponential(double scale) {
    return gsl_ran_exponential(r, scale);
}

double sample_normal() {
    return gsl_ran_gaussian(r, 1.0);
}

double sample_gamma(double shape, double scale) {
    return gsl_ran_gamma(r, shape, scale);
}

double sample_beta(double a, double b) {
    return gsl_ran_beta(r, a, b);
}

/* ---------------------------------------------------------------------
   Simulator: run an Euler integration SIR model over ndays.
   The arrays S, I, R must be preallocated with length ndays.
   This version uses local variables (rather than dynamic arrays) 
   to compute the infection fraction and new infections/recoveries.
   --------------------------------------------------------------------- */
void simulator(double beta, double gamma, int ndays, int N, int I0, int *S, int *I, int *R) {
    S[0] = N - I0;
    I[0] = I0;
    R[0] = 0;

    for (int t = 1; t < ndays; t++) {
        double frac_infected = (double) I[t - 1] / N;
        int new_infections = binomial(S[t - 1], 1 - exp(-beta * frac_infected));
        int new_recoveries = binomial(I[t - 1], gamma);

        S[t] = S[t - 1] - new_infections;
        I[t] = I[t - 1] + new_infections - new_recoveries;
        R[t] = R[t - 1] + new_recoveries;

        if (S[t] + I[t] + R[t] != N) {
            fprintf(stderr, "Error: S + I + R != N at time %d\n", t);
            exit(EXIT_FAILURE);
        }
    }
}

/* ---------------------------------------------------------------------
   Compute Euclidean norm difference between two integer arrays.
   --------------------------------------------------------------------- */
double compute_norm(const int *a, const int *b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

/* ---------------------------------------------------------------------
   Compute distances between observed and simulated data (S, I, R arrays).
   The distances are normalized by the total population N.
   --------------------------------------------------------------------- */
void distance(int N, int ndays, 
              const int *obs_S, const int *obs_I, const int *obs_R, 
              const int *sim_S, const int *sim_I, const int *sim_R,
              double *d_S, double *d_I, double *d_R) {
    *d_S = compute_norm(obs_S, sim_S, ndays) / N;
    *d_I = compute_norm(obs_I, sim_I, ndays) / N;
    *d_R = compute_norm(obs_R, sim_R, ndays) / N;
}

double compute_percentile(double *data, int n, double perc) {
    double *tmp = malloc(n * sizeof(double));
    if (tmp == NULL) {
        fprintf(stderr, "Error: failed to allocate memory in compute_percentile\n");
        exit(EXIT_FAILURE);
    }
    if (!tmp) {
        fprintf(stderr, "Error: failed to allocate memory in compute_percentile\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++)
        tmp[i] = data[i];
    gsl_sort(tmp, 1, n);
    double p = gsl_stats_quantile_from_sorted_data(tmp, 1, n, perc/100);
    free(tmp);
    return p;
}

void weight(int population_size, double beta_old, double beta_new, 
            double gamma_old, double gamma_new, 
            double exponent_scale, double beta_a, double beta_b,
            double w_beta_old, double w_gamma_old,
        sum_beta += w_beta_old * gsl_ran_gaussian_pdf(beta_old, GAUSSIAN_STD_DEV);
    double sum_beta = 0;
    double sum_gamma = 0;
    for (int j = 0; j < population_size; j++) {
        sum_beta += w_beta_old * gsl_ran_gaussian_pdf(beta_new, 0.01);
    }
    *w_beta = w_beta_old / sum_beta;

    for (int j = 0; j < population_size; j++) {
        sum_gamma += w_gamma_old * gsl_ran_gaussian_pdf(gamma_new, 0.01);
    }
    *w_gamma = w_gamma_old / sum_gamma;
}

// --- Parameters for SIR model and SMC-ABC ---
#define NDAYS 50
#define POPULATION 10000    // total SIR population
#define I0 10               // initial infections

// SMC-ABC settings
#define N_PARTICLES 1000    // number of particles per generation
#define N_GENERATIONS 5

// Priors parameters (example values)
#define EXPONENT_SCALE 1.0  // for exponential prior (for beta)
#define BETA_A 2.0          // for beta distribution (for gamma)
#define BETA_B 5.0
// Define standard deviation for Gaussian PDF in weight calculation
#define GAUSSIAN_STD_DEV 0.01

// Function for computing variance (simple sample variance)
void compute_variances(double *vals, int n, double *mean, double *var) {
#define SIM_SIZE NDAYS

// Function for computing variance (simple sample variance)
void compute_variances(double *vals, int n, double *mean, double *var) {
    double sum = 0, sum2 = 0;
    for(int i=0;i<n;i++){
        sum += vals[i];
        sum2 += vals[i]*vals[i];
    }
    *mean = sum/n;
    *var = (sum2 - (sum*sum)/n) / (n-1);
}

int main(){
    // Initialize GSL RNG
    gsl_rng_env_setup();
    r = gsl_rng_alloc(gsl_rng_default);
    srand(time(NULL));
    r = gsl_rng_alloc(gsl_rng_default);

    // Observed data arrays (simulate using fixed “true” parameters)
    int obs_S[SIM_SIZE], obs_I[SIM_SIZE], obs_R[SIM_SIZE];
    double beta_true = 0.3, gamma_true = 0.1;
    simulator(beta_true, gamma_true, NDAYS, POPULATION, I0, obs_S, obs_I, obs_R);

    // Allocate arrays for SMC-ABC (2D arrays, generations x particles)
    // Allocate arrays for SMC-ABC (2D arrays, generations x particles)
    double **beta_arr = malloc(N_GENERATIONS * sizeof(double*));
    double **gamma_arr = malloc(N_GENERATIONS * sizeof(double*));
    double *distS = malloc(N_PARTICLES * sizeof(double));
    if (distS == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for distS\n");
        exit(EXIT_FAILURE);
    }
    double *distI = malloc(N_PARTICLES * sizeof(double));
    if (distI == NULL) {
    // double *distR = malloc(N_PARTICLES * sizeof(double));
        exit(EXIT_FAILURE);
    }
    double *distR = malloc(N_PARTICLES * sizeof(double));
    if (distR == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for distR\n");
        exit(EXIT_FAILURE);
    if (!beta_arr || !gamma_arr || !distS || !distI || !distR) {
    double *distR = malloc(N_PARTICLES * sizeof(double));

    if (!beta_arr || !gamma_arr || !wgt_arr || !distS || !distI || !distR) {
        fprintf(stderr, "Error: failed to allocate memory for arrays\n");
        exit(EXIT_FAILURE);
    }

    for (int t = 0; t < N_GENERATIONS; t++){
        beta_arr[t] = malloc(N_PARTICLES * sizeof(double));
        gamma_arr[t] = malloc(N_PARTICLES * sizeof(double));
        wgt_arr[t]  = malloc(N_PARTICLES * sizeof(double));
        if (!beta_arr[t] || !gamma_arr[t] || !wgt_arr[t]) {
            fprintf(stderr, "Error: failed to allocate memory for arrays at generation %d\n", t);
            exit(EXIT_FAILURE);
        }
    }
    // Temporary simulation arrays
    int sim_S[SIM_SIZE], sim_I[SIM_SIZE], sim_R[SIM_SIZE];
    double dS, dI, dR;

    // ---------------- Generation 0 ----------------
    printf("Generation 0\n");
    for (int i = 0; i < N_PARTICLES; i++){
        // sample from priors
        beta_arr[0][i] = sample_exponential(EXPONENT_SCALE);
        gamma_arr[0][i] = sample_beta(BETA_A, BETA_B);
        // simulate model
        simulator(beta_arr[0][i], gamma_arr[0][i], NDAYS, POPULATION, I0, sim_S, sim_I, sim_R);
        // compute distances with observed data
        distance(POPULATION, NDAYS, obs_S, obs_I, obs_R, sim_S, sim_I, sim_R, &dS, &dI, &dR);
        distS[i] = dS; distI[i] = dI; distR[i] = dR;
        // all particles have equal initial weight
        wgt_arr[0][i] = 1.0 / N_PARTICLES;
    }

    // Compute tolerances (75th percentile on each distance)
    double tol_S = compute_percentile(distS, N_PARTICLES, 75.0);
    double tol_I = compute_percentile(distI, N_PARTICLES, 75.0);
    double tol_R = compute_percentile(distR, N_PARTICLES, 75.0);
    printf("Tolerance for generation 0: tol_S=%.3f tol_I=%.3f tol_R=%.3f\n", tol_S, tol_I, tol_R);

    // ---------------- Subsequent Generations ----------------
    for (int t = 1; t < N_GENERATIONS; t++){
        printf("Generation %d\n", t);
        // Compute simple variances for beta and gamma from previous generation (for perturbation)
        double mean_beta, var_beta, mean_gamma, var_gamma;
        compute_variances(beta_arr[t-1], N_PARTICLES, &mean_beta, &var_beta);
        compute_variances(gamma_arr[t-1], N_PARTICLES, &mean_gamma, &var_gamma);
        double sigma_beta = sqrt(var_beta);
        double sigma_gamma = sqrt(var_gamma);

        // For storing distances for the new generation
        for(int i=0;i<N_PARTICLES;i++){
            distS[i] = 0; distI[i] = 0; distR[i] = 0;
        }

        // For each particle in generation t, perform importance sampling/resampling & perturbation
        for (int i = 0; i < N_PARTICLES; i++){
            int accepted = 0;
            double proposed_beta, proposed_gamma;
            double curr_dS, curr_dI, curr_dR;
            int parent_index = -1;
            while (!accepted) {
                // Resample parent index from generation t-1 based on weights
                double u = gsl_rng_uniform(r);
                double cum = 0.0;
                for (int j = 0; j < N_PARTICLES; j++){
                    cum += wgt_arr[t-1][j];
                    if(u <= cum){
                        parent_index = j;
                        break;
                    }
                }

                proposed_beta = beta_arr[t-1][parent_index] + gsl_ran_gaussian(r, sigma_beta);
                proposed_gamma = gamma_arr[t-1][parent_index] + gsl_ran_gaussian(r, sigma_gamma);
                // Enforce non-negativity
                if (proposed_beta < 0) proposed_beta = 0;
                if (proposed_gamma < 0) proposed_gamma = 0;
                proposed_gamma = gamma_arr[t-1][parent_index] + gsl_ran_gaussian(rng, sigma_gamma);
                // Enforce non-negativity
                if (proposed_beta < 0) proposed_beta = 0;
                if (proposed_gamma < 0) proposed_gamma = 0;

                // Simulate model and compute distances
                simulator(proposed_beta, proposed_gamma, NDAYS, POPULATION, I0, sim_S, sim_I, sim_R);
                distance(POPULATION, NDAYS, obs_S, obs_I, obs_R, sim_S, sim_I, sim_R, &curr_dS, &curr_dI, &curr_dR);

                // Accept if all distances are below the previous tolerances
                if(curr_dS <= tol_S && curr_dI <= tol_I && curr_dR <= tol_R){
                    accepted = 1;
                    beta_arr[t][i] = proposed_beta;
                    gamma_arr[t][i] = proposed_gamma;
                    distS[i] = curr_dS;
                    distI[i] = curr_dI;
                    distR[i] = curr_dR;
                    // Compute weight using the provided weighting function
                    // Note: here we use the parent's parameter and weight (wgt_arr[t-1][parent_index])
                    double new_weight_beta, new_weight_gamma;
                    weight(N_PARTICLES, beta_arr[t-1][parent_index], proposed_beta, 
                           gamma_arr[t-1][parent_index], proposed_gamma, 
                           EXPONENT_SCALE, BETA_A, BETA_B,
                           wgt_arr[t-1][parent_index], wgt_arr[t-1][parent_index],
                           &new_weight_beta, &new_weight_gamma);
                    // For simplicity, combine the two weights (e.g. product)
                    wgt_arr[t][i] = new_weight_beta * new_weight_gamma;
                    break;
                } // else try new perturbation
            } // while accepted
        } // for each i

        // Normalize weights for generation t
        double sumw = 0.0;
        for (int i = 0; i < N_PARTICLES; i++){
            sumw += wgt_arr[t][i];
        }
        for (int i = 0; i < N_PARTICLES; i++){
            wgt_arr[t][i] /= sumw;
        }

        // Update tolerances using the 75th percentile from current distances
        tol_S = compute_percentile(distS, N_PARTICLES, 75.0);
        tol_I = compute_percentile(distI, N_PARTICLES, 75.0);
        tol_R = compute_percentile(distR, N_PARTICLES, 75.0);
        printf("Tolerance for generation %d: tol_S=%.3f tol_I=%.3f tol_R=%.3f\n", t, tol_S, tol_I, tol_R);
    }

    // {
    //         char filename[256];
    //         sprintf(filename, "/home/ubuntu/abc_sir/c_code/abc_smc_data/abc_smc_gen_last.csv");
    //         FILE *fp = fopen(filename, "w");
    //         if (fp) {
    //             fprintf(fp, "beta,gamma,epsilon\n");
    //             for (int i = 0; i < N_PARTICLES; i++) {
    //                 fprintf(fp, "%.6f,%.6f\n", beta_arr[N_GENERATIONS-1][i], gamma_arr[N_GENERATIONS-1][i]);
    //             }
    //             fclose(fp);
    //         }
    // }

    // // ---------------- Final posterior output ----------------
    // printf("Final posterior samples (last generation):\n");
    // for (int i = 0; i < N_PARTICLES; i++){
    //     printf("Sample %d: beta = %.3f, gamma = %.3f, weight = %.5f\n", 
    //             i, beta_arr[N_GENERATIONS-1][i], gamma_arr[N_GENERATIONS-1][i], wgt_arr[N_GENERATIONS-1][i]);
    for (int t = 0; t < N_GENERATIONS; t++){
        if (beta_arr[t]) free(beta_arr[t]);
        if (gamma_arr[t]) free(gamma_arr[t]);
        if (wgt_arr[t]) free(wgt_arr[t]);
    }
        free(gamma_arr[t]);
        free(wgt_arr[t]);
    }
    free(beta_arr);
    free(gamma_arr);
    free(wgt_arr);
    free(distS);
    free(distI);
    gsl_rng_free(r);
    gsl_rng_free(r);

    return 0;
}
}
// ...existing code if any...
