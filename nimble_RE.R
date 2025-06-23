rm(list = ls())
library(nimble)
library(coda)
library(splines)

J <- 60
prob_z <- 0.8
z <- rbinom(J, 1, prob_z)
tau_true <- rnorm(J, mean = 1.25 * z, sd = sqrt(1 / 12))
# A mixture distribution with mean 1 and var 1/3 (µ=1,σ=0.577)

se2 <- runif(J, min = 0.1, max = 0.2)
se <- sqrt(se2)
tau_hat <- rnorm(J, mean = tau_true, sd = se)

m <- 80 # grid length
degree <- 3 # spline degree
p <- 8 # degrees of freedom
grid <- seq(-1, 3, length.out = m)
Q <- bs(grid, df = p, degree = degree, intercept = FALSE)

code <- nimbleCode({
  for (k in 1:p) alpha[k] ~ dnorm(0, sd = 10)

  eta[1:m] <- Q[1:m, 1:p] %*% alpha[1:p]
  max_eta <- max(eta[1:m])
  exp_eta[1:m] <- exp(eta[1:m] - max_eta)
  sum_exp <- sum(exp_eta[1:m])
  p_i[1:m] <- exp_eta[1:m] / sum_exp

  for (j in 1:J) {
    z[j] ~ dcat(p_i[1:m])
    tau_hat[j] ~ dnorm(mean = grid[z[j]], sd = se[j])
  }

  mu <- inprod(grid[1:m], p_i[1:m])
  sqdiff[1:m] <- (grid[1:m] - mu)^2
  sigma <- sqrt(inprod(sqdiff[1:m], p_i[1:m]))
})

data <- list(
  tau_hat = tau_hat,
  grid    = grid
)

constants <- list(
  J = J,
  m = m,
  p = p,
  Q = Q,
  se = se
)

inits <- list(
  alpha = rep(0, p),
  z     = sample(1:m, J, replace = TRUE)
)

model <- nimbleModel(code,
  constants = constants,
  data      = data,
  inits     = inits,
  calculate = FALSE
)
c_model <- compileNimble(model)
conf <- configureMCMC(model, monitors = c("alpha", "mu", "sigma"))
r_mcmc <- buildMCMC(conf)
c_r_mcmc <- compileNimble(r_mcmc, project = model)

samples <- runMCMC(c_r_mcmc,
  niter = 2000,
  nburnin = 1000,
  samplesAsCodaMCMC = T
)
summary(samples)
