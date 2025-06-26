rm(list = ls())
library(rstan)
library(splines)
library(MASS)
options(mc.cores = parallel::detectCores())

# Mixture normal (u_j) + normal (v_ij) + correlated sampling error
J <- 200 # number of observations
G <- 60 # number of groups

# force each group to appear at least once
repeat {
  group <- sample(1:G, J, replace = TRUE)
  if (length(unique(group)) == G) break
}

prob_z <- 0.8
z <- rbinom(G, 1, prob_z)
u <- rnorm(G, mean = 1.25 * z, sd = sqrt(1 / 12))
v <- rnorm(J, mean = 0, sd = 1 / 2)
tau_true <- u[group] + v

se2 <- runif(J, min = 0.1, max = 0.2)
se <- sqrt(se2)
rho <- 0.75

error <- numeric(J)
J_arr <- tabulate(group, nbins = G)
J_max <- max(J_arr)
tau_hat_arr <- matrix(0, nrow = G, ncol = J_max)
V_arr <- array(0, dim = c(G, J_max, J_max))

for (g in 1:G) {
  idx <- which(group == g)
  se_g <- se[idx]
  Sigma_g <- outer(se_g, se_g, function(x, y) rho * x * y)
  diag(Sigma_g) <- se_g^2
  J_g <- J_arr[g]
  error[idx] <- mvrnorm(1, mu = rep(0, J_g), Sigma = Sigma_g)
  tau_hat_arr[g, 1:J_g] <- tau_true[idx] + error[idx]
  V_arr[g, 1:J_g, 1:J_g] <- Sigma_g
}

tau_hat <- tau_true + error

m <- 80
degree <- 3
p <- 8
grid <- seq(-1, 3, length.out = m)
Q <- bs(grid, df = p, degree = degree, intercept = FALSE)

stan_data <- list(
  J_max = J_max,
  G = G,
  m = m,
  p = p,
  J_arr = J_arr,
  tau_hat_arr = tau_hat_arr,
  V_arr = V_arr,
  Q = Q,
  grid = grid
)


model_code <- "data {
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
"

stan_mod <- stan_model(model_code = model_code)

fit_mcmc <- sampling(
  object = stan_mod,
  data = stan_data
)

print(fit_mcmc, pars = c("alpha", "mu", "sigma", "omega"))

fit_opt <- optimizing(
  object  = stan_mod,
  data    = stan_data,
  hessian = TRUE
)

fit_opt$par[grep("^alpha\\[", names(fit_opt$par))]
fit_opt$par["mu"]
fit_opt$par["sigma"]
fit_opt$par["omega"]
