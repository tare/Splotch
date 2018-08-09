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
  int<lower=1> N_spots[N_tissues]; // number of spots per tissue section
  int<lower=1> N_covariates; // number of AARs
  int<lower=1> N_genotypes; // number of genotypes
  int<lower=1> N_mice; // number of mice
  int<lower=1> N_sexes; // number of sexes
  int<lower=1> N_timepoints[N_genotypes]; // number of timepoints per genotype

  // genotype/timepoint/sex index of each mouse (this is used for indexing beta_sex)
  int<lower=1> mouse_mapping[N_mice]; 
  // mouse index of each tissue section (this is used for indexing beta_mouse)
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
  // cumulative sum of timepoints over genotypes (makes indexing a bit easier)
  int<lower=0> csum_N_timepoints[N_genotypes];
  // total number of spots
  int<lower=0> sum_N_spots;
  // total number of timepoints
  int<lower=0> sum_N_timepoints;

  // transform size factors to log space
  log_size_factors = log(size_factors);

  // calculate cumulative sum of spots over tissue sections
  csum_N_spots[1] = 0;
  for (i in 2:(N_tissues+1)) {
    csum_N_spots[i] = csum_N_spots[i-1]+N_spots[i-1];
  }

  // calculate cumulative sum of timepoints over genotypes 
  csum_N_timepoints[1] = 0;
  for (i in 2:N_genotypes) {
    csum_N_timepoints[i] = csum_N_timepoints[i-1]+N_timepoints[i-1];
  }

  // get total number of spots
  sum_N_spots = sum(N_spots);

  // get total number of timepoints
  sum_N_timepoints = sum(N_timepoints);
}

parameters {
  // CAR
  vector[sum_N_spots] psi;

  // non-centered parametrization of coefficients
  matrix[sum_N_timepoints+N_sexes*sum_N_timepoints+N_mice,N_covariates] beta_raw;

  // conditional precision
  real<lower=0> tau;
  // spatial autocorrelation
  real<lower=0,upper=1> a;

  // standard deviation of epsilon (spot-level variation)
  real<lower=0> sigma;

  // standard deviations of sex and mouse level in linear model
  real<lower=0> sigma_sex;
  real<lower=0> sigma_mouse;
  
  // probability of extra zeros
  real<lower=0,upper=1> theta;

  // non-centered parametrization of spot-level variation
  vector[sum_N_spots] noise_raw;
}

transformed parameters {
  // rate parameter
  vector[sum_N_spots] lambda;

  // genotype/timepoint level coefficients
  matrix[sum_N_timepoints,N_covariates] beta_genotype_timepoint;
  // sex level coefficients
  matrix[N_sexes*(sum_N_timepoints),N_covariates] beta_sex;
  // mouse level coefficients
  matrix[N_mice,N_covariates] beta_mouse;

  // derive genotype/timepoint level coefficients from beta_raw
  for (i in 1:N_genotypes) {
    for (j in 1:N_timepoints[i]) {
      beta_genotype_timepoint[csum_N_timepoints[i]+j] =
        2.0*beta_raw[csum_N_timepoints[i]+j];
    }
  }

  // derive sex level coefficients from beta_genotype_timepoint, sigma_sex, and beta_raw
  for (i in 1:N_genotypes) {
    for (j in 1:N_timepoints[i]) {
      for (k in 1:N_sexes) {
        beta_sex[csum_N_timepoints[i]*N_sexes+j+(k-1)*N_timepoints[i]] = 
          beta_genotype_timepoint[csum_N_timepoints[i]+j]
          +sigma_sex*beta_raw[csum_N_timepoints[i]*N_sexes+j
          +(k-1)*N_timepoints[i]+sum_N_timepoints];
      }
    }
  }

  // derive mouse level coefficients using beta_sex, sigma_mouse, and beta_raw
  for (i in 1:N_mice) {
    // get different mice from the genotype/timepoint/sex combinations
    // mouse_mapping is used to get correct beta_sex vector for each mouse 
    beta_mouse[i] = beta_sex[mouse_mapping[i]]
      +sigma_mouse*beta_raw[i+sum_N_timepoints+N_sexes*sum_N_timepoints];
  }

  // derive lambda using beta_mouse, psi, sigma, and noise_raw
  // D is used to get correct element from beta_mouse for each spot
  // tissue_mapping is used to get correct beta_mouse vector for each tissue section
  for (i in 1:N_tissues) {
    lambda[csum_N_spots[i]+1:csum_N_spots[i+1]] = 
      beta_mouse[tissue_mapping[i]][D[csum_N_spots[i]+1:csum_N_spots[i+1]]]'
      +psi[csum_N_spots[i]+1:csum_N_spots[i+1]]
      +0.3*sigma*noise_raw[csum_N_spots[i]+1:csum_N_spots[i+1]];
  }
}

model {
  // parameters of CAR (a has a Uniform(0,1) prior)
  tau ~ inv_gamma(1,1);

  // parameter of probability of extra zeros
  theta ~ beta(1,2);

  // CAR
  psi ~ sparse_car(tau,a,W_sparse,D_sparse,eig_values,sum_N_spots,W_n);

  // spot-level variation
  // non-centered parameterization
  sigma ~ normal(0,1);
  noise_raw ~ normal(0,1);

  // linear model
  // non-centered parameterization
  sigma_sex ~ normal(0,1);
  sigma_mouse ~ normal(0,1);
  to_vector(beta_raw) ~ normal(0,1);

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
