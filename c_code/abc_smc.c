#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>

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

// Perturb a sample given a list of accepted parameters.
Parameters perturb_sample(Parameters *accepted_samples, int count, double std_beta, double std_gamma, gsl_rng *rng) {
    Parameters new_par;
    int idx;
    do {
        // Randomly select a sample from the list of accepted samples.
        idx = gsl_rng_uniform_int(rng, count);
        double beta = accepted_samples[idx].beta;
        double gamma = accepted_samples[idx].gamma;
        new_par.beta = beta + gsl_ran_gaussian(rng, std_beta); // Gaussian random walk
        new_par.gamma = gamma + gsl_ran_gaussian(rng, std_gamma);
    }
    // Ensure beta > 0 and 0 <= gamma <= 1 
    while(new_par.beta <= 0 || new_par.gamma < 0 || new_par.gamma > 1);
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
    memcpy(copy, data, size*sizeof(double));
    qsort(copy, size, sizeof(double), compare_doubles);
    int idx = (int) floor(percentile/100.0 * (size - 1));
    double value = copy[idx];
    free(copy);
    return value;
}

// Main ABC SMC routine.
void run_abc(int n_particles, int n_generations, int ndays, int N, int I0, SimulationData *observed, gsl_rng *rng) {
    int gen, i;
    double tol[3] = {INFINITY, INFINITY, INFINITY}; // Initial tolerances for S, I, R
    Parameters *prev_accepted = NULL;  // previous generation accepted samples
    
    FILE *fp_stats = NULL; // File pointer for writing generation stats
    char filename_stats[256];
    snprintf(filename_stats, sizeof(filename_stats), "%sgeneration_stats.csv", PATH);
    fp_stats = fopen(filename_stats, "w");
    if(!fp_stats){
        fprintf(stderr, "Error opening %s for writing.\n", filename_stats);
        exit(EXIT_FAILURE);
    }
    fprintf(fp_stats, "generation,exec_time,tol_S,tol_I,tol_R\n");
    
    // Loop over generations
    for(gen = 0; gen < n_generations; gen++){
        clock_t start_time = clock();
        Parameters *accepted_params = malloc(n_particles * sizeof(Parameters));
        double *ds_arr = malloc(n_particles * sizeof(double));
        double *di_arr = malloc(n_particles * sizeof(double));
        double *dr_arr = malloc(n_particles * sizeof(double));
        int count = 0;
        
        // Sample util n_particles accepted samples are obtained.
        while(count < n_particles) {
            Parameters par;
            // Sample from prior for first generation, else perturb from previous accepted samples.
            if(gen == 0) {
                par = prior(0.2, 0.03, 1.0, rng);
            } else {
                /*// Estimate perturbation std from previous accepted samples.
                double sum_beta = 0, sum_gamma = 0;
                for(i = 0; i < n_particles; i++) {
                    sum_beta += prev_accepted[i].beta;
                    sum_gamma += prev_accepted[i].gamma;
                }
                double mean_beta = sum_beta / n_particles;
                double mean_gamma = sum_gamma / n_particles;
                double var_beta = 0, var_gamma = 0;
                for(i = 0; i < n_particles; i++) {
                    var_beta += pow(prev_accepted[i].beta - mean_beta, 2);
                    var_gamma += pow(prev_accepted[i].gamma - mean_gamma, 2);
                }
                double std_beta = (var_beta>0 ? sqrt(var_beta/n_particles) : 0.1);
                double std_gamma = (var_gamma>0 ? sqrt(var_gamma/n_particles) : 0.1);*/
                double *beta_array = malloc(n_particles * sizeof(double));
                double *gamma_array = malloc(n_particles * sizeof(double));
                for(i = 0; i < n_particles; i++){
                    beta_array[i] = prev_accepted[i].beta;
                    gamma_array[i] = prev_accepted[i].gamma;
                }
                double std_beta = gsl_stats_sd(beta_array, 1, n_particles);
                double std_gamma = gsl_stats_sd(gamma_array, 1, n_particles);
                // Perturb sample
                par = perturb_sample(prev_accepted, n_particles, std_beta, std_gamma, rng);
                free(beta_array);
                free(gamma_array);
            }
            SimulationData *sim = simulator(par.beta, par.gamma, ndays, N, I0, rng);
            Distances d = distance(N, observed, sim);
            free_simulation(sim);
            if(d.dS <= tol[0] && d.dI <= tol[1] && d.dR <= tol[2]) {
                accepted_params[count] = par;
                ds_arr[count] = d.dS;
                di_arr[count] = d.dI;
                dr_arr[count] = d.dR;
                count++;
            }
        }
        
        // Save accepted parameters for this generation to CSV.
        char filename[256];
        snprintf(filename, sizeof(filename), "%saccepted_params_gen_%d.csv", PATH, gen);
        FILE *fp = fopen(filename, "w");
        if(!fp){
            fprintf(stderr, "Error opening %s for writing.\n", filename);
            exit(EXIT_FAILURE);
        }
        fprintf(fp, "beta,gamma\n");
        for(i = 0; i < n_particles; i++){
            fprintf(fp, "%lf,%lf\n", accepted_params[i].beta, accepted_params[i].gamma);
        }
        fclose(fp);
        
        // Update tolerances using 75th percentile.
        tol[0] = compute_percentile(ds_arr, n_particles, 75.0);
        tol[1] = compute_percentile(di_arr, n_particles, 75.0);
        tol[2] = compute_percentile(dr_arr, n_particles, 75.0);
        printf("Generation %d completed. New tolerance values - dS: %lf, dI: %lf, dR: %lf\n", gen, tol[0], tol[1], tol[2]);
        
        double exec_time = ((double) (clock() - start_time)) / CLOCKS_PER_SEC;
        printf("Generation %d execution time: %.2lf seconds\n", gen, exec_time);
        fprintf(fp_stats, "%d,%.2lf,%lf,%lf,%lf\n", gen, exec_time, tol[0], tol[1], tol[2]);
        
        // Free previous accepted samples if exist.
        if(prev_accepted != NULL)
            free(prev_accepted);
        prev_accepted = accepted_params;
        free(ds_arr);
        free(di_arr);
        free(dr_arr);
    }
    fclose(fp_stats);
    if(prev_accepted)
        free(prev_accepted);
}

// Main function.
int main() {
    // Initialize RNG
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(rng, 9725); // set seed for reproducibility
    
    // Simulation settings
    int ndays = 600;    // number of days to simulate
    int N = 1000;       // total population
    int I0 = 10;        // initial infected
    
    // True parameters for observed data
    double true_beta = 0.1;
    double true_gamma = 0.01;
    SimulationData *observed = simulator(true_beta, true_gamma, ndays, N, I0, rng);
    
    // ABC SMC settings
    int n_particles = 10000;  // number of particles per generation
    int n_generations = 16;    // number of generations
    
    printf("Starting ABC SMC...\n");
    run_abc(n_particles, n_generations, ndays, N, I0, observed, rng);
    
    free_simulation(observed);
    gsl_rng_free(rng);
    return 0;
}