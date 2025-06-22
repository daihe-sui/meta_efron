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
for (g in seq_len(G)) {
  idx <- which(group == g)
  se_g <- se[idx]
  Sigma_g <- outer(se_g, se_g, function(x, y) rho * x * y)
  diag(Sigma_g) <- se_g^2
  error[idx] <- mvrnorm(1, rep(0, length(idx)), Sigma_g)
}

beta <- 0.5
q <- 1
X <- matrix(rep(0:1, each = J / 2), nrow = J, ncol = q)
tau_hat <- as.vector(X %*% beta) + tau_true + error

m <- 80
p <- 8
grid <- seq(-1, 3, length = m)
Q <- as.matrix(bs(grid, df = p, degree = 3, intercept = FALSE))

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
  grid = grid,
  q = q,
  X = X
)

# compile & fit

stan_mod <- stan_model(file = "efron_CHE_reg.stan")
fit_mcmc <- sampling(
  object = stan_mod,
  data = stan_data
)

print(fit_mcmc, pars = c("alpha", "mu", "sigma", "omega", "beta"))

fit_opt <- optimizing(
  object  = stan_mod,
  data    = stan_data,
  hessian = TRUE
)

fit_opt$par[grep("^alpha\\[", names(fit_opt$par))]
fit_opt$par["mu"]
fit_opt$par["sigma"]
fit_opt$par["omega"]
fit_opt$par["beta[1]"]
