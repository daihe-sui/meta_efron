data {
  int<lower=1> J_max;
  int<lower=1> G;
  int<lower=1> m;
  int<lower=1> p;
  int J_arr[G];
  vector[J_max] tau_hat_arr[G];
  matrix[J_max, J_max] V_arr[G];
  matrix[m, p] Q;
  vector[m] grid;
}

parameters {
  vector[p] alpha;
  real<lower=0> omega;
}

transformed parameters {
  vector[m] p_i = softmax(Q * alpha);
}

model {
  alpha ~ normal(0, 1);
  omega ~ cauchy(0, 5);

  vector[m] log_p_i = log(p_i);

  for (g in 1:G) {
    int J_g = J_arr[g];
    matrix[J_g, J_g] V_g = V_arr[g][1:J_g, 1:J_g]
      + diag_matrix(rep_vector(omega^2, J_g));
    vector[J_g] tau_hat_g = tau_hat_arr[g][1:J_g];
    vector[m] lps = log_p_i;

    for (i in 1:m) {
      lps[i] += multi_normal_lpdf(tau_hat_g | rep_vector(grid[i], J_g), V_g);
    }

    target += log_sum_exp(lps);
  }
}

generated quantities {
  real mu    = dot_product(grid, p_i);
  real sigma = sqrt(dot_product(p_i, square(grid - mu)));
}
