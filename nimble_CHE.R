rm(list = ls())
library(nimble)
library(coda)
library(splines)
library(MASS)

# Simulate data --- Mixture normal (u_j) + normal (v_ij) + correlated sampling error
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
v <- rnorm(J, mean = 0, sd = 0.5) # v_ij ~ N(0, ω=0.5)
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

code <- nimbleCode({
  for (k in 1:p) alpha[k] ~ dnorm(0, sd = 10)
  omega ~ dunif(0, 1000)

  eta[1:m] <- Q[1:m, 1:p] %*% alpha[1:p]
  max_eta <- max(eta[1:m])
  exp_eta[1:m] <- exp(eta[1:m] - max_eta)
  sum_exp <- sum(exp_eta[1:m])
  p_i[1:m] <- exp_eta[1:m] / sum_exp

  for (g in 1:G) {
    z[g] ~ dcat(p_i[1:m])
    for (j in 1:J_arr[g]) {
      v[g, j] ~ dnorm(0, sd = omega)
    }
    mu_arr[g, 1:J_arr[g]] <- grid[z[g]] + v[g, 1:J_arr[g]]
    tau_hat_arr[g, 1:J_arr[g]] ~ dmnorm(
      mean = mu_arr[g, 1:J_arr[g]],
      cov  = V_arr[g, 1:J_arr[g], 1:J_arr[g]]
    )
  }

  mu <- inprod(grid[1:m], p_i[1:m])
  sqdiff[1:m] <- (grid[1:m] - mu)^2
  sigma <- sqrt(inprod(sqdiff[1:m], p_i[1:m]))
})

data <- list(
  tau_hat_arr = tau_hat_arr,
  grid = grid
)
constants <- list(
  G = G,
  m = m,
  p = p,
  Q = Q,
  J_arr = J_arr,
  V_arr = V_arr
)
inits <- list(
  alpha = rep(0, p),
  z     = sample(1:m, G, replace = TRUE),
  v     = matrix(0, nrow = G, ncol = J_max),
  omega = 1
)

model <- nimbleModel(code,
  data      = data,
  constants = constants,
  inits     = inits,
  calculate = FALSE
)
c_model <- compileNimble(model)
conf <- configureMCMC(model, monitors = c("alpha", "mu", "sigma", "omega"))
r_mcmc <- buildMCMC(conf)
c_r_mcmc <- compileNimble(r_mcmc, project = model)

samples <- runMCMC(c_r_mcmc,
  niter             = 2000,
  nburnin           = 1000,
  samplesAsCodaMCMC = TRUE
)
summary(samples)
