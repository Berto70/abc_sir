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

/* ---------------------------------------------------------------------
   Compute the median of data (the temporary copy is sorted).
   --------------------------------------------------------------------- */
void compute_median(double *data, int n, double *median) {
    double *tmp = malloc(n * sizeof(double));
    if (!tmp) {
        fprintf(stderr, "Error: failed to allocate memory in compute_median\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++)
        tmp[i] = data[i];
    gsl_sort(tmp, 1, n);
    *median = gsl_stats_median_from_sorted_data(tmp, 1, n);
    free(tmp);
}

/* ---------------------------------------------------------------------
   Compute a given percentile of the data.
   --------------------------------------------------------------------- */
double compute_percentile(double *data, int n, double perc) {
    double *tmp = malloc(n * sizeof(double));
    if (!tmp) {
        fprintf(stderr, "Error: failed to allocate memory in compute_percentile\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++)
        tmp[i] = data[i];
    gsl_sort(tmp, 1, n);
    double p = gsl_stats_quantile_from_sorted_data(tmp, 1, n, perc);
    free(tmp);
    return p;
}

/* ---------------------------------------------------------------------
   Save particles and epsilon threshold to a CSV file.
   --------------------------------------------------------------------- */
void save_csv(const char *filename, double *betas, double *gammas, 
              double *epsilons_S, double *epsilons_I, double *epsilons_R, int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open file %s for writing.\n", filename);
        return;
    }
    fprintf(fp, "beta,gamma,epsilon_S,epsilon_I,epsilon_R\n");
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%.6f,%.6f,%.6f,%.6f,%.6f\n", betas[i], gammas[i], 
                epsilons_S[i], epsilons_I[i], epsilons_R[i]);
    }
    fclose(fp);
}

/* ---------------------------------------------------------------------
   Update particle weights.
   (Note: In the original code the loops simply summed a constant. 
   Here we simplify by computing the sum directly.)
   --------------------------------------------------------------------- */
// void weight(int population_size, double beta_old, double beta_new, 
//             double gamma_old, double gamma_new, 
//             double exponent_scale, double beta_a, double beta_b,
//             double w_beta_old, double w_gamma_old,
//             double *w_beta, double *w_gamma) {
//     double sum_beta = population_size * w_beta_old * gsl_ran_gaussian_pdf(beta_old, 0.01);
//     *w_beta = sample_exponential(exponent_scale) / sum_beta;

//     double sum_gamma = population_size * w_gamma_old * gsl_ran_gaussian_pdf(gamma_old, 0.01);
//     *w_gamma = sample_beta(beta_a, beta_b) / sum_gamma;
// }

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

/* ---------------------------------------------------------------------
   Sample an index from an array of weights.
   --------------------------------------------------------------------- */
int weighted_sample(gsl_rng *rng, double *weights, int npart) {
    double r_uniform = gsl_rng_uniform(rng);
    double cum_sum = 0.0;
    for (int i = 0; i < npart; i++) {
        cum_sum += weights[i];
        if (r_uniform <= cum_sum)
            return i;
    }
    return npart - 1;  // Fallback to last element if necessary.
}

/* --------------------------------------------------------------------- 
Run the ABC-SMC algorithm. For generation 0 we sample from the prior until 
we have population_size particles. For subsequent generations we use importance 
resampling with perturbation. 
--------------------------------------------------------------------- */ 
void run_abc_smc(const int *obs_S, const int *obs_I, const int *obs_R, int ndays, 
                int N, int I0, int n_generations, int population_size, 
                double exponent_scale, double beta_a, double beta_b) { 
                    // Allocate arrays for particles. 
                double *particles_beta = malloc(population_size * sizeof(double)); 
                double *particles_gamma = malloc(population_size * sizeof(double)); 
                if (!particles_beta || !particles_gamma) { 
                    fprintf(stderr, "Error: failed to allocate memory for particles.\n"); 
                    exit(EXIT_FAILURE); }

    // Allocate weight arrays for each generation.
    double (*weight_beta)[n_generations + 1] = malloc(population_size * sizeof(double[n_generations + 1]));
    double (*weight_gamma)[n_generations + 1] = malloc(population_size * sizeof(double[n_generations + 1]));
    if (!weight_beta || !weight_gamma) {
        fprintf(stderr, "Error: failed to allocate memory for weight arrays.\n");
        exit(EXIT_FAILURE);
    }

    // Allocate epsilon arrays for each generation (generation 0 to n_generations).
    double *epsilon_S = malloc((n_generations + 1) * sizeof(double));
    double *epsilon_I = malloc((n_generations + 1) * sizeof(double));
    double *epsilon_R = malloc((n_generations + 1) * sizeof(double));
    if (!epsilon_S || !epsilon_I || !epsilon_R) {
        fprintf(stderr, "Error: failed to allocate memory for epsilon arrays.\n");
        exit(EXIT_FAILURE);
    }

    // --- Generation 0: sample from the prior ---
    {
        printf("Running Generation 0 ...\n");
        double *distances_S = malloc(population_size * sizeof(double));
        double *distances_I = malloc(population_size * sizeof(double));
        double *distances_R = malloc(population_size * sizeof(double));
        if (!distances_S || !distances_I || !distances_R) {
            fprintf(stderr, "Error: failed to allocate memory for distance arrays in generation 0.\n");
            exit(EXIT_FAILURE);
        }
        int accepted = 0;
        while (accepted < population_size) {
            double beta = sample_exponential(exponent_scale);
            double gamma = sample_beta(beta_a, beta_b);
            // Check if beta and gamma are within boundaries
            if (beta <= 0 || gamma <= 0 || gamma > 1)
                continue;

            int *sim_S = malloc(ndays * sizeof(int));
            int *sim_I = malloc(ndays * sizeof(int));
            int *sim_R = malloc(ndays * sizeof(int));
            if (!sim_S || !sim_I || !sim_R) {
                fprintf(stderr, "Error: failed to allocate memory for simulation arrays.\n");
                exit(EXIT_FAILURE);
            }

            simulator(beta, gamma, ndays, N, I0, sim_S, sim_I, sim_R);
            double d_S, d_I, d_R;
            distance(N, ndays, obs_S, obs_I, obs_R, sim_S, sim_I, sim_R, &d_S, &d_I, &d_R);

            particles_beta[accepted] = beta;
            particles_gamma[accepted] = gamma;
            weight_beta[accepted][0] = 1.0;
            weight_gamma[accepted][0] = 1.0;
            distances_S[accepted] = d_S;
            distances_I[accepted] = d_I;
            distances_R[accepted] = d_R;

            free(sim_S);
            free(sim_I);
            free(sim_R);

            accepted++;
        }
        // Compute epsilon thresholds for generation 0 (75th percentiles).
        epsilon_S[0] = compute_percentile(distances_S, population_size, 0.75);
        epsilon_I[0] = compute_percentile(distances_I, population_size, 0.75);
        epsilon_R[0] = compute_percentile(distances_R, population_size, 0.75);
        printf("Generation 0: epsilon = %.5f, %.5f, %.5f\n", epsilon_S[0], epsilon_I[0], epsilon_R[0]);

        char filename[256];
        sprintf(filename, "/home/ubuntu/abc_sir/c_code/abc_smc_data/particles_gen_%d.csv", 0);
        save_csv(filename, particles_beta, particles_gamma, &epsilon_S[0], &epsilon_I[0], &epsilon_R[0], population_size);

        free(distances_S);
        free(distances_I);
        free(distances_R);
    }

    // --- Subsequent generations: importance resampling with perturbation ---
    for (int gen = 1; gen <= n_generations; gen++) {
        printf("Running Generation %02d ...\n", gen);
        double *distances_S = malloc(population_size * sizeof(double));
        double *distances_I = malloc(population_size * sizeof(double));
        double *distances_R = malloc(population_size * sizeof(double));
        if (!distances_S || !distances_I || !distances_R) {
            fprintf(stderr, "Error: failed to allocate memory for distance arrays in generation %d.\n", gen);
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < population_size; i++) {
            while (1) {
                // Sample an index for beta from the previous generation weights.
                double r_uniform = gsl_rng_uniform(r);
                double cum_sum = 0.0;
                int idx_beta = -1;
                for (int j = 0; j < population_size; j++) {
                    cum_sum += weight_beta[j][gen-1];
                    if (r_uniform <= cum_sum) {
                        idx_beta = j;
                        break;
                    }
                }
                if (idx_beta == -1)
                    idx_beta = population_size - 1;

                // Sample an index for gamma from the previous generation weights.
                r_uniform = gsl_rng_uniform(r);
                cum_sum = 0.0;
                int idx_gamma = -1;
                for (int j = 0; j < population_size; j++) {
                    cum_sum += weight_gamma[j][gen-1];
                    if (r_uniform <= cum_sum) {
                        idx_gamma = j;
                        break;
                    }
                }
                if (idx_gamma == -1)
                    idx_gamma = population_size - 1;

                double old_beta = particles_beta[idx_beta];
                double old_gamma = particles_gamma[idx_gamma];

                // Perturb the parameters.
                double new_beta = old_beta + gsl_ran_gaussian(r, 0.01);
                double new_gamma = old_gamma + gsl_ran_gaussian(r, 0.01);

                // Check if beta and gamma are within boundaries.
                if (new_beta <= 0 || new_gamma <= 0 || new_gamma > 1)
                    continue;

                int *sim_S = malloc(ndays * sizeof(int));
                int *sim_I = malloc(ndays * sizeof(int));
                int *sim_R = malloc(ndays * sizeof(int));
                if (!sim_S || !sim_I || !sim_R) {
                    fprintf(stderr, "Error: failed to allocate memory for simulation arrays.\n");
                    exit(EXIT_FAILURE);
                }
                simulator(new_beta, new_gamma, ndays, N, I0, sim_S, sim_I, sim_R);
                double d_S, d_I, d_R;
                distance(N, ndays, obs_S, obs_I, obs_R, sim_S, sim_I, sim_R, &d_S, &d_I, &d_R);
                free(sim_S);
                free(sim_I);
                free(sim_R);

                // Accept the new particle if it meets the previous generation's thresholds.
                if (d_S < epsilon_S[gen - 1] && d_I < epsilon_I[gen - 1] && d_R < epsilon_R[gen - 1]) {
                    particles_beta[i] = new_beta;
                    particles_gamma[i] = new_gamma;
                    weight(population_size, old_beta, new_beta, old_gamma, new_gamma,
                        exponent_scale, beta_a, beta_b,
                        weight_beta[idx_beta][gen - 1], weight_gamma[idx_gamma][gen - 1],
                        &weight_beta[i][gen], &weight_gamma[i][gen]);
                    distances_S[i] = d_S;
                    distances_I[i] = d_I;
                    distances_R[i] = d_R;
                    break;
                }
                // If not accepted, try again (the allocated simulation arrays have been freed).
            }
        }
        // Compute new epsilon thresholds for this generation.
        epsilon_S[gen] = compute_percentile(distances_S, population_size, 0.75);
        epsilon_I[gen] = compute_percentile(distances_I, population_size, 0.75);
        epsilon_R[gen] = compute_percentile(distances_R, population_size, 0.75);

        printf("Generation %02d: epsilon = %.5f, %.5f, %.5f\n", gen, epsilon_S[gen], epsilon_I[gen], epsilon_R[gen]);

        char filename[256];
        sprintf(filename, "/home/ubuntu/abc_sir/c_code/abc_smc_data/particles_gen_%d.csv", gen);
        save_csv(filename, particles_beta, particles_gamma, &epsilon_S[gen], &epsilon_I[gen], &epsilon_R[gen], population_size);

        free(distances_S);
        free(distances_I);
        free(distances_R);
    }

    // Free allocated arrays.
    free(particles_beta);
    free(particles_gamma);
    free(weight_beta);
    free(weight_gamma);
    free(epsilon_S);
    free(epsilon_I);
    free(epsilon_R);

}

/* --------------------------------------------------------------------- 
Main function. 
--------------------------------------------------------------------- */ 
int main(void) { 
    // Initialize the GSL random number generator. 
    const gsl_rng_type *T; 
    gsl_rng_env_setup(); 
    T = gsl_rng_default; 
    r = gsl_rng_alloc(T); 
    if (!r) { fprintf(stderr, "Error: failed to allocate GSL random number generator.\n"); 
    exit(EXIT_FAILURE); }

    clock_t start_t = clock();

    // Problem parameters.
    int ndays = 600, N = 1000, I0 = 10;
    double fiducial_beta = 0.1, fiducial_gamma = 0.01;
    int population_size = 10000;      // Number of particles.
    int n_generations = 5;

    // Prior parameters.
    double exponent_scale = 0.1, beta_a = 0.01, beta_b = 1.0;

    // Allocate observed data arrays.
    int *obs_S = malloc(ndays * sizeof(int));
    int *obs_I = malloc(ndays * sizeof(int));
    int *obs_R = malloc(ndays * sizeof(int));
    if (!obs_S || !obs_I || !obs_R) {
        fprintf(stderr, "Error: failed to allocate memory for observed data.\n");
        exit(EXIT_FAILURE);
    }

    // Create a mock dataset using fiducial parameters.
    simulator(fiducial_beta, fiducial_gamma, ndays, N, I0, obs_S, obs_I, obs_R);

    // Run the ABC-SMC algorithm.
    run_abc_smc(obs_S, obs_I, obs_R, ndays, N, I0, n_generations, population_size,
                exponent_scale, beta_a, beta_b);

    clock_t end_t = clock();
    double exec_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("Total Execution time: %.3f seconds\n", exec_t);

    // Free allocated memory.
    free(obs_S);
    free(obs_I);
    free(obs_R);
    gsl_rng_free(r);

    return 0;


}