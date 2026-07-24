# Résumé agrégé de l'étude thesis-v2

Les intervalles des méthodes apprises sont des IC Student-t à 95 % sur les seeds d'entraînement.
Les intervalles des baselines sont calculés sur les mondes d'évaluation.

## Méthodes

| Scope | pD | Méthode | Politique | Seeds | Reward | Completed | Coverage | Collision |
|---|---:|---|---|---:|---:|---:|---:|---:|
| multi | 0.50 | baseline_local_frontier | deterministic | 1 | 8.359 | 3.438 | 0.849 | 0.423 |
| multi | 0.70 | baseline_local_frontier | deterministic | 1 | 10.696 | 3.584 | 0.819 | 0.409 |
| multi | 0.50 | baseline_random | deterministic | 1 | -7.831 | 0.149 | 0.611 | 0.005 |
| multi | 0.70 | baseline_random | deterministic | 1 | -7.980 | 0.149 | 0.611 | 0.005 |
| multi | 0.50 | bayes_qmix_independent | posterior_mean | 4 | -7.074 | 0.385 | 0.585 | 0.063 |
| multi | 0.70 | bayes_qmix_independent | posterior_mean | 4 | -7.403 | 0.342 | 0.555 | 0.066 |
| multi | 1.00 | bayes_qmix_independent | posterior_mean | 3 | -7.934 | 0.365 | 0.616 | 0.062 |
| multi | 0.50 | bayes_qmix_independent | posterior_sample_independent | 4 | -7.414 | 0.337 | 0.571 | 0.076 |
| multi | 0.70 | bayes_qmix_independent | posterior_sample_independent | 4 | -7.628 | 0.318 | 0.547 | 0.077 |
| multi | 1.00 | bayes_qmix_independent | posterior_sample_independent | 3 | -8.281 | 0.302 | 0.580 | 0.078 |
| multi | 0.50 | bayes_qmix_independent | posterior_sample_shared | 4 | -7.753 | 0.314 | 0.546 | 0.121 |
| multi | 0.70 | bayes_qmix_independent | posterior_sample_shared | 4 | -7.791 | 0.313 | 0.531 | 0.104 |
| multi | 1.00 | bayes_qmix_independent | posterior_sample_shared | 3 | -8.358 | 0.316 | 0.557 | 0.114 |
| multi | 1.00 | bayes_qmix_shared | posterior_mean | 4 | -7.143 | 0.299 | 0.498 | 0.040 |
| multi | 1.00 | bayes_qmix_shared | posterior_sample_independent | 4 | -7.429 | 0.296 | 0.506 | 0.052 |
| multi | 1.00 | bayes_qmix_shared | posterior_sample_shared | 4 | -7.460 | 0.287 | 0.493 | 0.067 |
| multi | 0.50 | qmix_ddqn | deterministic | 4 | -6.999 | 0.407 | 0.600 | 0.096 |
| multi | 0.70 | qmix_ddqn | deterministic | 4 | -6.885 | 0.436 | 0.576 | 0.090 |
| multi | 1.00 | qmix_ddqn | deterministic | 4 | -7.895 | 0.337 | 0.592 | 0.100 |
| multi | 1.00 | shared_ddqn | deterministic | 4 | -5.871 | 0.238 | 0.400 | 0.058 |
| single | 0.50 | baseline_frontier | deterministic | 1 | 11.179 | 3.458 | 0.882 | nan |
| single | 0.70 | baseline_frontier | deterministic | 1 | 13.047 | 3.660 | 0.845 | nan |
| single | 0.50 | baseline_oracle | deterministic | 1 | 20.528 | 4.000 | 0.357 | nan |
| single | 0.70 | baseline_oracle | deterministic | 1 | 20.549 | 4.000 | 0.357 | nan |
| single | 0.50 | baseline_random | deterministic | 1 | -3.059 | 0.055 | 0.277 | nan |
| single | 0.70 | baseline_random | deterministic | 1 | -3.122 | 0.055 | 0.277 | nan |
| single | 1.00 | bdqn | posterior_mean | 4 | -2.874 | 0.094 | 0.242 | nan |
| single | 1.00 | bdqn | posterior_sample | 4 | -2.913 | 0.056 | 0.214 | nan |
| single | 1.00 | ddqn | deterministic | 4 | -3.355 | 0.076 | 0.240 | nan |

## Comparaisons appariées

| pD | Comparaison | Paires | Δ reward | Δ completed | Δ coverage | Δ collision |
|---:|---|---:|---:|---:|---:|---:|
| 0.50 | bayes_qmix_independent_mean_minus_qmix | 4 | -0.075 | -0.023 | -0.015 | -0.032 |
| 0.70 | bayes_qmix_independent_mean_minus_qmix | 4 | -0.518 | -0.093 | -0.021 | -0.024 |
| 1.00 | bayes_qmix_independent_mean_minus_qmix | 3 | -0.195 | 0.021 | 0.023 | -0.035 |
| 0.50 | bayes_qmix_independent_sample_minus_qmix | 4 | -0.415 | -0.070 | -0.029 | -0.020 |
| 0.70 | bayes_qmix_independent_sample_minus_qmix | 4 | -0.743 | -0.117 | -0.029 | -0.013 |
| 1.00 | bayes_qmix_independent_sample_minus_qmix | 3 | -0.542 | -0.042 | -0.012 | -0.019 |
| 1.00 | bayes_qmix_shared_mean_minus_qmix | 4 | 0.752 | -0.039 | -0.094 | -0.060 |
| 1.00 | bayes_qmix_shared_sample_minus_qmix | 4 | 0.435 | -0.050 | -0.099 | -0.033 |
| 1.00 | bdqn_mean_minus_ddqn | 4 | 0.481 | 0.018 | 0.001 | nan |
| 1.00 | bdqn_sample_minus_ddqn | 4 | 0.442 | -0.020 | -0.026 | nan |
| 1.00 | independent_sampling_minus_shared_sampling | 3 | -0.943 | 0.016 | 0.088 | 0.012 |
| 1.00 | qmix_minus_shared_ddqn | 4 | -2.024 | 0.099 | 0.192 | 0.043 |
