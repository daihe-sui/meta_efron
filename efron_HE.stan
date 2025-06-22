data {
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

