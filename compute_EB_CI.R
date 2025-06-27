rm(list = ls())
library(rstan)
library(splines)

J <- 200
z <- rbinom(J, size = 1, prob = 0.8)
tau_true <- rnorm(J, mean = 5 / 4 * z, sd = sqrt(1 / 12))
se2_j <- runif(J, min = 0.2, max = 0.5)
tau_hat <- rnorm(J, mean = tau_true, sd = sqrt(se2_j))
se_j <- sqrt(se2_j)

m <- 80
degree <- 3
p <- 8
grid <- seq(-1, 3, length.out = m)
Q <- bs(grid, df = p, degree = degree, intercept = T)

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
    vector[J]       tau_hat;
    vector<lower=0>[J] se;
    int<lower=1> m;
    int<lower=1> p;
    vector[m]       grid;
    matrix[m, p] Q;
}
parameters {
    vector[p] alpha;
}
transformed parameters {
    simplex[m] w = softmax(Q * alpha);
}
model {
    alpha ~ normal(0, 1);
    for (j in 1:J) {
        vector[m] lps = log(w);
        for (i in 1:m) {
            lps[i] += normal_lpdf(tau_hat[j] | grid[i], se[j]);
        }
        target += log_sum_exp(lps);
    }
}
generated quantities {
    real mu     = dot_product(grid, w);
    real sigma = sqrt(dot_product(w, square(grid - mu)));
    vector[J] tau_eb;
      for (j in 1:J) {
        vector[m] lps = log(w);
        for (i in 1:m) {
          lps[i] += normal_lpdf(tau_hat[j] | grid[i], se[j]);
        }
        vector[m] w_j = softmax(lps);
        tau_eb[j] = dot_product(grid, w_j);
      }
}
"

stan_mod <- stan_model(model_code = model_code)

fit_opt <- optimizing(
  object  = stan_mod,
  data    = stan_data,
  hessian = TRUE
)

# Calculating CIs for mu and sigma

alpha <- fit_opt$par[grep("^alpha\\[", names(fit_opt$par))]
hessian <- fit_opt$hessian

# Sigma_alpha = -H(alpha)^-1
Sigma_alpha <- -solve(hessian)

eta <- as.vector(Q %*% alpha)
w <- exp(eta - max(eta))
w <- w / sum(w)
mu <- fit_opt$par["mu"]
sigma <- fit_opt$par["sigma"]

# eta = Q %*% alpha
# d(w)/d(alpha) = d(w)/d(eta) * d(eta)/d(alpha)
# d(w)/d(eta) = diag(w) - w * t(w) (softmax Jacobian)
dw_deta <- diag(w) - w %*% t(w)
dw_dalpha <- dw_deta %*% Q

# mu = t(grid) %*% w
# d(mu)/d(alpha) = d(mu)/d(w) * d(w)/d(alpha) = t(grid) %*% d(w)/d(alpha)
grad_mu <- t(grid) %*% dw_dalpha

# sigma^2 = (t(grid^2) %*% w) - mu^2
# d(sigma^2)/d(alpha) = t(grid^2) %*% d(w)/d(alpha) - 2 * mu * d(mu)/d(alpha)
grad_var <- t(grid^2) %*% dw_dalpha - 2 * mu * grad_mu

# d(sigma)/d(alpha) = (1 / (2 * sigma)) * d(sigma^2)/d(alpha)
grad_sigma <- (1 / (2 * sigma)) * grad_var

# The Delta Method formula for the variance of mu and sigma
var_mu <- grad_mu %*% Sigma_alpha %*% t(grad_mu)
se_mu <- sqrt(var_mu)
var_sigma <- grad_sigma %*% Sigma_alpha %*% t(grad_sigma)
se_sigma <- sqrt(var_sigma)
ci_mu <- c(mu - 1.96 * se_mu, mu + 1.96 * se_mu)
ci_sigma <- c(sigma - 1.96 * se_sigma, sigma + 1.96 * se_sigma)
cat("95% CI for mu:", ci_mu, "\n")
cat("95% CI for sigma:", ci_sigma, "\n")

# Calculating CIs for tau_eb estimators

tau_eb <- fit_opt$par[grep("^tau_eb\\[", names(fit_opt$par))]
tau_eb_ci <- matrix(NA, nrow = J, ncol = 2)
colnames(tau_eb_ci) <- c("ci_lower", "ci_upper")

for (j in 1:J) {
  lps_j <- log(w) + dnorm(tau_hat[j], mean = grid, sd = se_j[j], log = TRUE)
  w_j <- exp(lps_j - max(lps_j))
  w_j <- w_j / sum(w_j)

  # d(w_j)/d(w) = d(w_j)/d(lps_j) * d(lps_j)/d(w)
  # d(w_j)/d(lps_j) = diag(w_j) - w_j * t(w_j) (softmax Jacobian)
  # d(lps_j)/d(w) = 1/w

  dw_j_dw <- (diag(w_j) * (1 / w)) - (w_j %*% t(w_j / w))

  # tau_eb_j = t(grid) %*% w_j
  # d(tau_eb_j)/d(w) = t(grid) %*% d(w_j)/d(w)
  dtau_eb_j_dw <- t(grid) %*% dw_j_dw

  # d(tau_eb_j)/d(alpha) = d(tau_eb_j)/d(w) * d(w)/d(alpha)
  grad_tau_eb_j <- dtau_eb_j_dw %*% dw_dalpha

  # The Delta Method formula for the variance of tau_eb_j
  var_tau_eb_j <- grad_tau_eb_j %*% Sigma_alpha %*% t(grad_tau_eb_j)
  se_tau_eb_j <- sqrt(var_tau_eb_j)
  tau_eb_ci[j, ] <- c(
    tau_eb[j] - 1.96 * se_tau_eb_j,
    tau_eb[j] + 1.96 * se_tau_eb_j
  )
}

results <- data.frame(
  tau_hat = tau_hat,
  tau_eb_estimate = tau_eb,
  ci_lower = tau_eb_ci[, 1],
  ci_upper = tau_eb_ci[, 2]
)

print(head(results), digits = 3)
