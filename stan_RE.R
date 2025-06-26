rm(list = ls())
library(rstan)
library(splines)
options(mc.cores = parallel::detectCores())

J <- 200
z <- rbinom(J, size = 1, prob = 0.8)
tau_true <- rnorm(J, mean = 5 / 4 * z, sd = sqrt(1 / 12))
se2_j <- runif(J, min = 0.2, max = 0.5)
tau_hat <- rnorm(J, mean = tau_true, sd = sqrt(se2_j))
se_j <- sqrt(se2_j)

m <- 80 # grid length
degree <- 3 # spline degree
p <- 8 # degrees of freedom
grid <- seq(-1, 3, length.out = m)
Q <- bs(grid, df = p, degree = degree, intercept = FALSE)

stan_data <- list(
  J       = J,
  tau_hat = tau_hat,
  se      = se_j,
  m       = m,
  p       = p,
  grid    = grid,
  Q       = Q
)

model_code <- "data {
    int<lower=1> J;
    vector[J]    tau_hat;
    vector<lower=0>[J] se;
    int<lower=1> m;
    int<lower=1> p;
    vector[m]    grid;
    matrix[m, p] Q;
}

parameters {
    vector[p] alpha;
}

transformed parameters {
    simplex[m] p_i = softmax(Q * alpha);
}

model {
    alpha ~ normal(0, 1);
    vector[m] log_p_i = log(p_i);
    for (j in 1:J) {
        vector[m] lps = log_p_i;
        for (i in 1:m) {
            lps[i] += normal_lpdf(tau_hat[j] | grid[i], se[j]);
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

print(fit_mcmc, pars = c("alpha", "mu", "sigma"))

fit_opt <- optimizing(
  object  = stan_mod,
  data    = stan_data,
  hessian = TRUE
)

fit_opt$par[grep("^alpha\\[", names(fit_opt$par))]
fit_opt$par["mu"]
fit_opt$par["sigma"]
