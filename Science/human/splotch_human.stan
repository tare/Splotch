functions {
  // sparse_car_lpdf is written by Max Joseph
  // see: http://mc-stan.org/users/documentation/case-studies/mbjoseph-CARStan.html
  real sparse_car_lpdf(vector phi, real tau, real alpha, int[,] W_sparse, vector D_sparse, vector lambda, int n, int W_n) {
    row_vector[n] phit_D;
    row_vector[n] phit_W;
    vector[n] ldet_terms;

    phit_D = (phi .* D_sparse)';
    phit_W = rep_row_vector(0,n);
    for (i in 1:W_n) {
      phit_W[W_sparse[i,1]] = phit_W[W_sparse[i,1]]+phi[W_sparse[i,2]];
      phit_W[W_sparse[i,2]] = phit_W[W_sparse[i,2]]+phi[W_sparse[i,1]];
    }
    ldet_terms = log1m(alpha*lambda);
    return 0.5*(n*log(tau)+sum(ldet_terms)-tau*(phit_D*phi-alpha*(phit_W*phi)));
  }
}

data {
  int<lower=1> N_tissues; // number of tissue sections
  int<lower=0> N_spots[N_tissues]; // number of spots per tissue section
  int<lower=1> N_covariates; // number of AARs
  int<lower=1> N_onsets_tissue_locations; // number of onset/location combinations
  int<lower=1> N_donors; // number of donor and tissue location combinations

  // onset/tissue location index of each donor (this is used for indexing beta_onset_location)
  int<lower=1> donor_mapping[N_donors];
  // donor index of each tissue section (this is used for indexing beta_donor_onset_tissue) 
  int<lower=1> tissue_mapping[N_tissues];

  int<lower=0> counts[sum(N_spots)]; // counts per spot

  vector<lower=0>[sum(N_spots)] size_factors; // size factor for each spot

  // annotation for each spot (this is used for indexing beta_mouse)
  int<lower=1> D[sum(N_spots)];

  int<lower=0> W_n; // number of adjacent spot pairs
  int W_sparse[W_n,2]; // adjacency pairs
  vector[sum(N_spots)] D_sparse; // number of neighbors for each spot
  vector[sum(N_spots)] eig_values; // eigenvalues of D^{-0.5} W D^{-0.5}
}

transformed data {
  // log-transformed size factors are more convenient (poisson_log)
  vector[sum(N_spots)] log_size_factors;
  // cumulative sum of spots over tissue sections (makes indexing a bit easier)
  int<lower=0> csum_N_spots[N_tissues+1];
  // total number of spots
  int<lower=0> sum_N_spots;
  
  // transform size factors to log space
  log_size_factors = log(size_factors);
  
  // calculate cumulative sum of spots over tissue sections
  csum_N_spots[1] = 0;
  for (i in 2:(N_tissues+1)) { 
    csum_N_spots[i] = csum_N_spots[i-1]+N_spots[i-1];
  }

  // get total number of spots
  sum_N_spots = sum(N_spots);
}

parameters {
  // CAR
  vector[sum_N_spots] psi;

  // non-centered parametrization of coefficients
  matrix[N_onsets_tissue_locations+N_donors,N_covariates] beta_raw;

  // conditional precision
  real<lower=0> tau;
  // spatial autocorrelation
  real<lower=0,upper=1> alpha;

  // standard deviation of epsilon (spot-level variation)
  real<lower=0> sigma;

  real<lower=0> sigma_donor;

  // probability of extra zeros
  real<lower=0,upper=1> theta;

  // non-centered parameterization of spot-level variation
  vector[sum_N_spots] noise_raw;
}

transformed parameters {
  // rate parameter
  vector[sum_N_spots] lambda;

  // onset/location level coefficients
  matrix[N_onsets_tissue_locations,N_covariates] beta_onset_tissue;
  // donor/tissue level coefficients
  matrix[N_donors,N_covariates] beta_donor_onset_tissue;

  // derive onset/location level coefficients from beta_raw
  for (i in 1:N_onsets_tissue_locations) {
    beta_onset_tissue[i] = 2.0*beta_raw[i];
  }

  // derive donor/onset/tissue level coefficients using beta_onset_location, sigma_donor, and beta_raw
  for (i in 1:N_donors) {
    // get different donorss from the onset/location combinations
    // donor_mapping is used to get correct beta_onset_location vector for each donor 
    beta_donor_onset_tissue[i] = beta_onset_tissue[donor_mapping[i]]
      +sigma_donor*beta_raw[i+N_onsets_tissue_locations];
  }

  // derive lambda using beta_donor_onset_tissue, psi, sigma, and noise_raw
  // D is used to get correct element from beta_donor_onset_tissue for each spot
  // tissue_mapping is used to get correct beta_mouse vector for each tissue section
  for (i in 1:N_tissues) {
    lambda[csum_N_spots[i]+1:csum_N_spots[i+1]] =
      beta_donor_onset_tissue[tissue_mapping[i]][D[csum_N_spots[i]+1:csum_N_spots[i+1]]]'
      +psi[csum_N_spots[i]+1:csum_N_spots[i+1]]
      +0.3*sigma*noise_raw[csum_N_spots[i]+1:csum_N_spots[i+1]];
  }
}

model {
  // parameters of CAR (a has a Uniform(0,1) prior)
  tau ~ inv_gamma(1,1);

  // parameter of probability of extra zeros
  theta ~ beta(1,2);

  // linear model
  // non centered parameterization
  sigma_donor ~ normal(0,1);
  to_vector(beta_raw) ~ normal(0,1);

  // spot-level variation
  // non-centered parameterization
  noise_raw ~ normal(0,1);
  sigma ~ normal(0,1);

  // CAR
  psi ~ sparse_car(tau,alpha,W_sparse,D_sparse,eig_values,sum_N_spots,W_n);

  // zero-inflated Poisson likelihood
  for (i in 1:sum_N_spots) {
    if (counts[i] == 0)
      target += log_sum_exp(bernoulli_lpmf(1|theta),
                            bernoulli_lpmf(0|theta)
                              +poisson_log_lpmf(counts[i]|lambda[i]+log_size_factors[i]));
    else
      target += bernoulli_lpmf(0|theta)
                  + poisson_log_lpmf(counts[i]|lambda[i]+log_size_factors[i]);
  }
}

generated quantities {
}
