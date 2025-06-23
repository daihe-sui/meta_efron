rm(list = ls())
library(nimble)
library(coda)
library(splines)

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
# u_j is a mixture normal with mean 1 and variance 1/3 (µ=1,σ=0.577)
v <- rnorm(J, 0, 0.5) # v_ij ~ N(0, ω=0.5)
tau_true <- u[group] + v

se2 <- runif(J, min = 0.1, max = 0.2)
se <- sqrt(se2)
tau_hat <- rnorm(J, mean = tau_true, sd = se)

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
  }

  for (j in 1:J) {
    u[j] ~ dnorm(0, sd = omega)
    tau_hat[j] ~ dnorm(
      mean = grid[z[group[j]]] + u[j],
      sd = se[j]
    )
  }

  mu <- inprod(grid[1:m], p_i[1:m])
  sqdiff[1:m] <- (grid[1:m] - mu)^2
  sigma <- sqrt(inprod(sqdiff[1:m], p_i[1:m]))
})

# 5. Prepare data & constants
data <- list(
  tau_hat = tau_hat,
  grid    = grid
)

constants <- list(
  J     = J,
  m     = m,
  p     = p,
  Q     = Q,
  se    = se,
  G     = G,
  group = group
)

inits <- list(
  alpha = rep(0, p),
  z     = sample(1:m, G, replace = TRUE),
  u     = rep(0, J),
  omega = 1
)

model <- nimbleModel(code,
  constants = constants,
  data      = data,
  inits     = inits,
  calculate = FALSE
)
c_model <- compileNimble(model)
conf <- configureMCMC(model,
  monitors = c("alpha", "mu", "sigma", "omega")
)
r_mcmc <- buildMCMC(conf)
c_r_mcmc <- compileNimble(r_mcmc, project = model)

samples <- runMCMC(c_r_mcmc,
  niter = 2000,
  nburnin = 1000,
  samplesAsCodaMCMC = T
)

post_means <- colMeans(as.matrix(samples))
alpha <- post_means[grep("^alpha\\[", names(post_means))]
mu <- post_means["mu"]
sigma <- post_means["sigma"]
omega <- post_means["omega"]
p_i <- exp(Q %*% alpha)
p_i <- p_i / sum(p_i)

par(mfrow = c(2, 1))
hist(tau_hat, breaks = 40)
plot(grid - mu, p_i / diff(grid)[1], type = "l", xlab = "", ylab = "Density")
lines(grid - mu, dnorm(grid - mu, mean = 0, sd = omega), col = "red")
legend("topright",
  legend = c("Between groups", "Within group"),
  col = c("black", "red"),
  lty = 1
)
title(main = paste0("mu=", round(mu, 4)))

summary(samples)
