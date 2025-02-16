#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

using namespace std;

// Global random engine.
std::random_device rd;
std::mt19937 rng(rd());

struct SIRResult {
    vector<int> S;
    vector<int> I;
    vector<int> R;
};

SIRResult simulator(double beta, double gamma, int ndays, int N, int I0 = 10) {
    SIRResult result;
    result.S.resize(ndays);
    result.I.resize(ndays);
    result.R.resize(ndays);
    vector<double> frac_infected(ndays, 0.0);

    result.S[0] = N - I0;
    result.I[0] = I0;
    result.R[0] = 0;
    frac_infected[0] = static_cast<double>(I0) / N;

    for (int t = 1; t < ndays; t++) {
        // new infections and recoveries
        double p_infection = 1 - exp(-beta * frac_infected[t - 1]);
        std::binomial_distribution<int> binom_infect(result.S[t-1], p_infection);
        int dI = binom_infect(rng);

        std::binomial_distribution<int> binom_recover(result.I[t-1], gamma);
        int dR = binom_recover(rng);

        result.S[t] = result.S[t-1] - dI;
        result.I[t] = result.I[t-1] + dI - dR;
        result.R[t] = result.R[t-1] + dR;

        frac_infected[t] = static_cast<double>(result.I[t]) / N;

        if (result.S[t] + result.I[t] + result.R[t] != N) {
            throw std::runtime_error("The sum of S, I, and R should be equal to the total population size");
        }
    }
    return result;
}

double norm_diff(const vector<int>& a, const vector<int>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

struct Distance {
    double d_S;
    double d_I;
    double d_R;
};

Distance distanceCalc(int N, const SIRResult& observed, const SIRResult& simulated) {
    Distance d;
    d.d_S = norm_diff(simulated.S, observed.S) / N;
    d.d_I = norm_diff(simulated.I, observed.I) / N;
    d.d_R = norm_diff(simulated.R, observed.R) / N;
    return d;
}

// Sample beta ~ Exponential(1) with given scale (mean = exponent_scale)
double sample_beta(double exponent_scale) {
    std::exponential_distribution<double> exp_dist(1.0 / exponent_scale);
    return exp_dist(rng);
}

// Sample gamma ~ Beta(a,b) using two gamma distributions.
double sample_gamma(double a, double b) {
    std::gamma_distribution<double> gamma_dist_a(a, 1.0);
    std::gamma_distribution<double> gamma_dist_b(b, 1.0);
    double X = gamma_dist_a(rng);
    double Y = gamma_dist_b(rng);
    return X / (X + Y);
}

vector<double> accepted_betas;
vector<double> accepted_gammas;

void run_ABC(const SIRResult& observed,
             double exponent_scale, double beta_a, double beta_b,
             int ndays, int N, int n_iterations, double epsilon_fixed) {

    // vector<double> accepted_betas;
    // vector<double> accepted_gammas;
    vector<int> trials_per_iteration;
    int total_proposals = 0;

    float time_in = clock();

    for (int iter = 0; iter < n_iterations; iter++) {
        int trials = 0;
        while (true) {
            trials++;
            total_proposals++;

            double beta = sample_beta(exponent_scale);
            double gamma = sample_gamma(beta_a, beta_b);

            SIRResult sim = simulator(beta, gamma, ndays, N);
            Distance d = distanceCalc(N, observed, sim);

            // if (iter % 100 == 0) {
            //     cout << "Iteration " << iter
            //          << " d_S: " << d.d_S 
            //          << ", d_I: " << d.d_I 
            //          << ", d_R: " << d.d_R << endl;
            // }

            if ((d.d_S <= epsilon_fixed) &&
                (d.d_I <= epsilon_fixed) &&
                (d.d_R <= epsilon_fixed)) {
                accepted_betas.push_back(beta);
                accepted_gammas.push_back(gamma);
                trials_per_iteration.push_back(trials);
                break;
            }
        }
    }
    // Compute average number of trials per accepted sample.
    double sum_trials = 0;
    for (auto t : trials_per_iteration) {
        sum_trials += t;
    }
    double avg_trials = sum_trials / trials_per_iteration.size();
    double acceptance_rate = static_cast<double>(accepted_betas.size()) / total_proposals;

    cout << "Acceptance rate: " << acceptance_rate 
         << ", Average number of trials: " << avg_trials << endl;

    float time_out = clock();
    cout << "Time: " << (time_out - time_in) / CLOCKS_PER_SEC << " seconds" << endl;
}

void print_median(const vector<double>& v) {
    vector<double> sorted = v;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    if (n % 2 == 0) {
        cout << "Median: " << (sorted[n/2 - 1] + sorted[n/2]) / 2 << endl;
    } else {
        cout << "Median: " << sorted[n/2] << endl;
    }
}

int main() {
    // For demonstration purposes, simulate an observed dataset 
    // using fixed parameters.
    int ndays = 600;
    int N = 1000;
    int I0 = 10;
    double true_beta = 0.1;
    double true_gamma = 0.01;
    
    SIRResult observed = simulator(true_beta, true_gamma, ndays, N, I0);

    // ABC parameters.
    double exponent_scale = 0.1; // mean for beta exponential
    double beta_a = 0.01;   // parameters for gamma sampling (for gamma ~ Beta)
    double beta_b = 1.0;
    int n_iterations = 100000;
    double epsilon_fixed = 20;  // fixed epsilon threshold

    run_ABC(observed, exponent_scale, beta_a, beta_b, ndays, N, n_iterations, epsilon_fixed);
    print_median(accepted_betas);
    print_median(accepted_gammas);
    return 0;
}