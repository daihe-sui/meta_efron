data {
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

