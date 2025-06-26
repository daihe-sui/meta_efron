rm(list = ls())
library(rstan)
library(splines)
library(MASS)
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

rho <- 0.75
error <- numeric(J)
for (g in 1:G) {
  idx <- which(group == g)
  se_g <- se[idx]
  Sigma_g <- outer(se_g, se_g, function(x, y) rho * x * y)
  diag(Sigma_g) <- se_g^2
  error[idx] <- mvrnorm(1, rep(0, length(idx)), Sigma_g)
}
tau_hat <- tau_true + error

m <- 80
p <- 8
degree <- 3
grid <- seq(-1, 3, length = m)
Q <- bs(grid, df = p, degree = degree, intercept = FALSE)

stan_data <- list(
  J = J,
  G = G,
  m = m,
  p = p,
  group = group,
  tau_hat = tau_hat,
  se = se,
  rho = rho,
  Q = Q,
  grid = grid
)

model_code <- "data {
    int<lower=1> J;
    int<lower=1> G;
    int<lower=1> m;
    int<lower=1> p;
    int<lower=1,upper=G> group[J];
    vector[J] tau_hat;
    vector[J] se;
    real<lower=0,upper=1> rho;
    matrix[m, p] Q;
    vector[m] grid;
}

transformed data {
    int J_arr[G];
    vector[J] tau_hat_arr[G];
    vector[J] se_arr[G];

    for (g in 1:G) {
        J_arr[g] = 0;
        for (j in 1:J) {
            if (group[j] == g) {
                J_arr[g] += 1;
                tau_hat_arr[g][J_arr[g]] = tau_hat[j];
                se_arr[g][J_arr[g]] = se[j];
            }
        }
    }
}

parameters {
    vector[p] alpha;
    real<lower=0> omega;
}

transformed parameters {
    simplex[m] p_i = softmax(Q * alpha);
}

model {
    alpha ~ normal(0, 1);
    omega ~ cauchy(0, 5);

    vector[m] log_p_i = log(p_i);

    for (g in 1:G) {
        int J_g = J_arr[g];
        vector[J_g] tau_g = tau_hat_arr[g][1:J_g];
        vector[J_g] se_g  = se_arr[g][1:J_g];

        vector[J_g] d  = (1 - rho) * square(se_g) + square(omega);
        vector[J_g] w  = 1 ./ d;
        vector[J_g] u  = sqrt(rho) * se_g;

        real t1 = dot_product(tau_g, w);
        real t2 = dot_product(tau_g .* w, tau_g);
        real u1 = dot_product(u, w);
        real u2 = dot_product(u .* w, u);
        real ut = dot_product(u .* w, tau_g);

        real log_det = sum(log(d)) + log1p(u2);

        vector[m] quad = rep_vector(t2, m)
            - 2 * t1 * grid
            + sum(w) * square(grid)
            - square(ut - u1 * grid) / (1 + u2);

        target += log_sum_exp(
            log_p_i
            - 0.5 * (J_g * log(2 * pi()) + log_det + quad)
        );
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
