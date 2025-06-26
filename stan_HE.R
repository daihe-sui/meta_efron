rm(list = ls())
library(rstan)
library(splines)
options(mc.cores = parallel::detectCores())

J <- 200
G <- 60
repeat {
  group <- sample(1:G, J, replace = TRUE)
  if (length(unique(group)) == G) break
}
prob_z <- 0.8
z <- rbinom(G, 1, prob_z)
u <- rnorm(G, mean = 1.25 * z, sd = sqrt(1 / 12))
v <- rnorm(J, 0, 0.5)
tau_true <- u[group] + v

se2 <- runif(J, 0.1, 0.2)
se <- sqrt(se2)
tau_hat <- rnorm(J, tau_true, se)

m <- 80
p <- 8
degree <- 3
grid <- seq(-1, 3, length.out = m)
Q <- bs(grid, df = p, degree = degree, intercept = FALSE)

stan_data <- list(
  J       = J,
  G       = G,
  m       = m,
  p       = p,
  group   = group,
  tau_hat = tau_hat,
  se      = se,
  grid    = grid,
  Q       = Q
)

model_code <- "data {
    int<lower=1>            J;
    int<lower=1>            G;
    int<lower=1>            m;
    int<lower=1>            p;
    int<lower=1,upper=G>    group[J];
    vector[J]               tau_hat;
    vector[J]               se;
    vector[m]               grid;
    matrix[m, p]            Q;
}

parameters {
    vector[p]               alpha;
    real<lower=0>           omega;
}

transformed parameters {
    simplex[m]              p_i = softmax(Q * alpha);
}

model {
    alpha ~ normal(0, 1);
    omega ~ cauchy(0, 5);
    vector[m]               log_p_i = log(p_i);
    matrix[m, G]            lps = rep_matrix(log_p_i, G);
    for (j in 1:J) {
        int   idx  = group[j];
        real  sd_j = sqrt(square(se[j]) + square(omega));
        for (i in 1:m) {
            lps[i, idx] += normal_lpdf(tau_hat[j] | grid[i], sd_j);
        }
    }
    for (g in 1:G) {
        target += log_sum_exp(lps[, g]);
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
  data = stan_data,
  chains = 4
)

print(fit_mcmc,
  pars = c("alpha", "mu", "sigma", "omega")
)

fit_opt <- optimizing(
  object  = stan_mod,
  data    = stan_data,
  hessian = TRUE
)

fit_opt$par[grep("^alpha\\[", names(fit_opt$par))]
fit_opt$par["mu"]
fit_opt$par["sigma"]
fit_opt$par["omega"]
