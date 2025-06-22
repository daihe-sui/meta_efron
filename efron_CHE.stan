data {
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
