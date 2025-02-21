/*
 * SIR Model Simulation
 * Author: Gabriele Bertinelli
 * Date: 21-02-2025
 *
 * This program simulates the spread of an infectious disease using the SIR model,
 * using the Sequential Monte Carlo ABC algorithm to infer the parameters of the model.
 * It uses a more correct version, calculating the importance weights.
 * It gives similar results to the base version of the same algorithm abc_smc.c.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
// #include <gsl/gsl_sf_beta.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PATH "/home/ubuntu/abc_sir/c_code/abc_smc_data/10k_0.2_0.03/"

// Data structure to store SIR simulation results.
typedef struct {
    int *S;
    int *I;
    int *R;
    int ndays;
} SimulationData;

// Data structure to store model parameters.
typedef struct {
    double beta;
    double gamma;
} Parameters;

// Data structure to store Euclidean distances.
typedef struct {
    double dS;
    double dI;
    double dR;
} Distances;

// Helper function to free SimulationData
void free_simulation(SimulationData *sim) {
    if(sim) {
        free(sim->S);
        free(sim->I);
        free(sim->R);
        free(sim);
    }
}

// SIR simulator: simulates for ndays days.
SimulationData* simulator(double beta, double gamma, int ndays, int N, int I0, gsl_rng *rng) {
    SimulationData *sim = malloc(sizeof(SimulationData));
    sim->ndays = ndays;
    sim->S = calloc(ndays, sizeof(int));
    sim->I = calloc(ndays, sizeof(int));
    sim->R = calloc(ndays, sizeof(int));
    
    double *frac_infected = calloc(ndays, sizeof(double));  // temporary array for fraction infected
    int t;
    
    sim->S[0] = N - I0;
    sim->I[0] = I0;
    sim->R[0] = 0;
    frac_infected[0] = (double)I0 / N;
    
    for(t = 1; t < ndays; t++){
        unsigned int s_prev = sim->S[t-1];
        unsigned int i_prev = sim->I[t-1];
        
        double p_inf = 1 - exp(-beta * frac_infected[t-1]);
        // Get new infections and recoveries using binomial draws
        unsigned int new_inf = gsl_ran_binomial(rng, p_inf, s_prev);
        unsigned int new_rec = gsl_ran_binomial(rng, gamma, i_prev);
        
        sim->S[t] = sim->S[t-1] - new_inf;
        sim->I[t] = sim->I[t-1] + new_inf - new_rec;
        sim->R[t] = sim->R[t-1] + new_rec;
        
        frac_infected[t] = (double)sim->I[t] / N;
        
        if(sim->S[t] + sim->I[t] + sim->R[t] != N){
            fprintf(stderr, "Error: S+I+R != N at day %d\n", t);
            free_simulation(sim);
            free(frac_infected);
            exit(EXIT_FAILURE);
        }
    }
    free(frac_infected);
    return sim;
}

// Prior sampler: β ~ Exponential(scale), γ ~ Beta(a, b)
Parameters prior(double exponent_scale, double beta_a, double beta_b, gsl_rng *rng) {
    Parameters par;
    par.beta = gsl_ran_exponential(rng, exponent_scale);
    par.gamma = gsl_ran_beta(rng, beta_a, beta_b);
    return par;
}

// Compute Euclidean norm difference normalized by N.
Distances distance(int N, SimulationData *observed, SimulationData *simulated) {
    Distances dist = {0.0, 0.0, 0.0};
    int t;
    double sum_S = 0, sum_I = 0, sum_R = 0;
    for(t = 0; t < observed->ndays; t++){
        sum_S += pow(simulated->S[t] - observed->S[t], 2);
        sum_I += pow(simulated->I[t] - observed->I[t], 2);
        sum_R += pow(simulated->R[t] - observed->R[t], 2);
    }
    dist.dS = sqrt(sum_S) / N;
    dist.dI = sqrt(sum_I) / N;
    dist.dR = sqrt(sum_R) / N;
    return dist;
}

// Gaussian PDF helper.
double gaussian_pdf(double x, double sigma) {
    return (1.0 / (sqrt(2 * M_PI) * sigma)) * exp(-0.5 * pow(x / sigma, 2));
}

// Prior density for beta ~ Exponential(scale).
double prior_beta_pdf(double beta, double exponent_scale, gsl_rng *rng) {
    return (1.0 / exponent_scale) * exp(-beta / exponent_scale);
}

// Prior density for gamma ~ Beta(beta_a, beta_b).
double prior_gamma_pdf(double gamma, double beta_a, double beta_b, gsl_rng *rng) {
    if (gamma < 0 || gamma > 1) return 0.0;
    double beta_pdf = tgamma(beta_a + beta_b) / (tgamma(beta_a) * tgamma(beta_b));
    return beta_pdf * pow(gamma, beta_a - 1) * pow(1 - gamma, beta_b - 1);
}

// Joint prior density: π(θ) = π(beta)*π(gamma)
double prior_joint_pdf(double beta, double gamma, double exponent_scale, double beta_a, double beta_b, gsl_rng *rng) {
    return prior_beta_pdf(beta, exponent_scale, rng) * prior_gamma_pdf(gamma, beta_a, beta_b, rng);
}

// Perturb kernel density: Gaussian density for each parameter.
double perturb_kernel_density(double new_beta, double new_gamma,
                              double prev_beta, double prev_gamma,
                              double std_beta, double std_gamma) {
    double d_beta = new_beta - prev_beta;
    double d_gamma = new_gamma - prev_gamma;
    return gaussian_pdf(d_beta, std_beta) * gaussian_pdf(d_gamma, std_gamma);
}

// Calculate the weight for a new particle.
// If generation == 0, weight is 1. Otherwise, apply the weighted formula.
double calculate_weight(Parameters new_sample,
                        Parameters *prev_samples, double *prev_weights, int n_prev,
                        double exponent_scale, double beta_a, double beta_b,
                        double std_beta, double std_gamma,
                        int generation) {
    if (generation == 0) {
        return 1.0;
    }
    
    double numerator = prior_joint_pdf(new_sample.beta, new_sample.gamma, exponent_scale, beta_a, beta_b, NULL);
    double denominator = 0.0;
    for (int j = 0; j < n_prev; j++) {
        double kernel = perturb_kernel_density(new_sample.beta, new_sample.gamma,
                                                 prev_samples[j].beta, prev_samples[j].gamma,
                                                 std_beta, std_gamma);
        denominator += prev_weights[j] * kernel;
    }
    if (denominator == 0.0)
        return 0.0;
    return numerator / denominator;
}

// Perturb a sample given a list of accepted parameters.
Parameters perturb_sample(Parameters *prev_samples, double *prev_weights, int count,
                            double std_beta, double std_gamma,
                            double exponent_scale, double beta_a, double beta_b,
                            gsl_rng *rng) {
    int idx;
    if (prev_weights == NULL) {
        // Generation 0: use uniform sampling.
        idx = gsl_rng_uniform_int(rng, count);
    } else {
        // Generation > 0: weighted resampling using previous weights.
        double total_weight = 0.0;
        for (int i = 0; i < count; i++) {
            total_weight += prev_weights[i];
        }
        double r = gsl_rng_uniform(rng) * total_weight;
        double cumulative = 0.0;
        for (int i = 0; i < count; i++) {
            cumulative += prev_weights[i];
            if (r <= cumulative) {
                idx = i;
                break;
            }
        }
    }
    
    Parameters new_par;
    // Perturb the selected previous particle until valid parameters are obtained.
    do {
        new_par.beta = prev_samples[idx].beta + gsl_ran_gaussian(rng, std_beta);
        new_par.gamma = prev_samples[idx].gamma + gsl_ran_gaussian(rng, std_gamma);
    } while (new_par.beta <= 0 || new_par.gamma < 0 || new_par.gamma > 1);
    
    return new_par;
}

// Helper function for double comparison in qsort.
int compare_doubles(const void *a, const void *b) {
    double diff = *(double*)a - *(double*)b;
    return (diff > 0) - (diff < 0);
}

// Compute the percentile value from an array of doubles.
double compute_percentile(double *data, int size, double percentile) {
    double *copy = malloc(size * sizeof(double));
    memcpy(copy, data, size * sizeof(double));
    qsort(copy, size, sizeof(double), compare_doubles);
    int idx = (int) floor(percentile / 100.0 * (size - 1));
    double value = copy[idx];
    free(copy);
    return value;
}

// Main ABC SMC routine.
void run_abc(int n_particles, int n_generations, int ndays, int N, int I0, SimulationData *observed, gsl_rng *rng) {
    int gen, i;
    double tol[3] = {INFINITY, INFINITY, INFINITY}; // Initial tolerances for S, I, R

    // Prior parameters for ABC SMC.
    double exponent_scale = 0.2;
    double beta_a = 0.03;
    double beta_b = 1.0;
    
    Parameters *prev_accepted = NULL;  // previous generation accepted samples
    double *prev_weights = NULL;       // previous generation weights

    FILE *fp_stats = NULL; // File pointer for writing generation stats
    char filename_stats[256];
    snprintf(filename_stats, sizeof(filename_stats), "%sgeneration_stats.csv", PATH);
    fp_stats = fopen(filename_stats, "w");
    if (!fp_stats) {
        fprintf(stderr, "Error opening %s for writing.\n", filename_stats);
        exit(EXIT_FAILURE);
    }
    fprintf(fp_stats, "generation,exec_time,tol_S,tol_I,tol_R\n");
    
    // Loop over generations
    for(gen = 0; gen < n_generations; gen++){
        clock_t start_time = clock();
        Parameters *accepted_params = malloc(n_particles * sizeof(Parameters));
        double *accepted_weights = malloc(n_particles * sizeof(double));
        double *ds_arr = malloc(n_particles * sizeof(double));
        double *di_arr = malloc(n_particles * sizeof(double));
        double *dr_arr = malloc(n_particles * sizeof(double));
        int count = 0;
        
        // For generation > 0, compute perturbation standard deviations once.
        double std_beta = 0.0, std_gamma = 0.0;
        if (gen > 0) {
            double *beta_array = malloc(n_particles * sizeof(double));
            double *gamma_array = malloc(n_particles * sizeof(double));
            for(i = 0; i < n_particles; i++){
                beta_array[i] = prev_accepted[i].beta;
                gamma_array[i] = prev_accepted[i].gamma;
            }
            std_beta = gsl_stats_sd(beta_array, 1, n_particles);
            std_gamma = gsl_stats_sd(gamma_array, 1, n_particles);
            free(beta_array);
            free(gamma_array);
        }
        
        // Sample until n_particles accepted samples are obtained.
        while(count < n_particles) {
            Parameters par;
            if(gen == 0) {
                par = prior(exponent_scale, beta_a, beta_b, rng);
            } else {
                par = perturb_sample(prev_accepted, prev_weights, n_particles, std_beta, std_gamma,
                                     exponent_scale, beta_a, beta_b, rng);
            }
            SimulationData *sim = simulator(par.beta, par.gamma, ndays, N, I0, rng);
            Distances d = distance(N, observed, sim);
            free_simulation(sim);
            
            if(d.dS <= tol[0] && d.dI <= tol[1] && d.dR <= tol[2]) {
                accepted_params[count] = par;
                ds_arr[count] = d.dS;
                di_arr[count] = d.dI;
                dr_arr[count] = d.dR;
                if (gen == 0) {
                    accepted_weights[count] = 1.0;
                } else {
                    double w = calculate_weight(par, prev_accepted, prev_weights, n_particles,
                                                exponent_scale, beta_a, beta_b, std_beta, std_gamma, gen);
                    accepted_weights[count] = w;
                }
                count++;
            }
        }
        
        // Save accepted parameters for this generation to CSV.
        char filename[256];
        snprintf(filename, sizeof(filename), "%saccepted_params_gen_%d.csv", PATH, gen);
        FILE *fp = fopen(filename, "w");
        if (!fp) {
            fprintf(stderr, "Error opening %s for writing.\n", filename);
            exit(EXIT_FAILURE);
        }
        fprintf(fp, "beta,gamma\n");
        for(i = 0; i < n_particles; i++){
            fprintf(fp, "%lf,%lf\n", accepted_params[i].beta, accepted_params[i].gamma);
        }
        fclose(fp);
        
        // Update tolerances using the 75th percentile.
        tol[0] = compute_percentile(ds_arr, n_particles, 75.0);
        tol[1] = compute_percentile(di_arr, n_particles, 75.0);
        tol[2] = compute_percentile(dr_arr, n_particles, 75.0);
        printf("Generation %d completed. New tolerance values - dS: %lf, dI: %lf, dR: %lf\n",
               gen, tol[0], tol[1], tol[2]);
        
        double exec_time = ((double)(clock() - start_time)) / CLOCKS_PER_SEC;
        printf("Generation %d execution time: %.2lf seconds\n", gen, exec_time);
        fprintf(fp_stats, "%d,%.2lf,%lf,%lf,%lf\n", gen, exec_time, tol[0], tol[1], tol[2]);
        
        // Free previous generation's accepted samples and weights.
        if(prev_accepted) free(prev_accepted);
        if(prev_weights) free(prev_weights);
        prev_accepted = accepted_params;
        prev_weights = accepted_weights;
        free(ds_arr);
        free(di_arr);
        free(dr_arr);
    }
    fclose(fp_stats);
    
    if(prev_accepted) free(prev_accepted);
    if(prev_weights) free(prev_weights);
}

// Main function.
int main() {
    // Initialize RNG.
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(rng, 9725); // set seed for reproducibility
    
    // Simulation settings.
    int ndays = 600;    // number of days to simulate.
    int N = 1000;       // total population.
    int I0 = 10;        // initial infected.
    
    // True parameters for observed data.
    double true_beta = 0.2;
    double true_gamma = 0.03;
    SimulationData *observed = simulator(true_beta, true_gamma, ndays, N, I0, rng);
    
    // ABC SMC settings.
    int n_particles = 10000;  // number of particles per generation.
    int n_generations = 16;    // number of generations.
    
    printf("Starting ABC SMC...\n");
    run_abc(n_particles, n_generations, ndays, N, I0, observed, rng);
    
    free_simulation(observed);
    gsl_rng_free(rng);
    return 0;
}
