# Résumé agrégé de l'étude thesis-v2

Les intervalles des méthodes apprises sont des IC Student-t à 95 % sur les seeds d'entraînement.
Les intervalles des baselines sont calculés sur les mondes d'évaluation.

## Méthodes

| Scope | pD | Méthode | Politique | Seeds | Reward | Completed | Coverage | Collision |
|---|---:|---|---|---:|---:|---:|---:|---:|
| multi | 0.70 | baseline_local_frontier | deterministic | 1 | 11.531 | 3.653 | 0.823 | 0.382 |
| multi | 1.00 | baseline_local_frontier | deterministic | 1 | 14.786 | 3.800 | 0.802 | 0.325 |
| multi | 0.70 | baseline_random | deterministic | 1 | -7.906 | 0.130 | 0.606 | 0.005 |
| multi | 1.00 | baseline_random | deterministic | 1 | -8.053 | 0.130 | 0.606 | 0.005 |
| multi | 0.70 | bayes_qmix_independent | posterior_mean | 3 | -8.085 | 0.312 | 0.634 | 0.067 |
| multi | 1.00 | bayes_qmix_independent | posterior_mean | 3 | -8.599 | 0.312 | 0.650 | 0.068 |
| multi | 0.70 | bayes_qmix_independent | posterior_sample_independent | 3 | -8.053 | 0.296 | 0.603 | 0.075 |
| multi | 1.00 | bayes_qmix_independent | posterior_sample_independent | 3 | -8.857 | 0.229 | 0.597 | 0.095 |
| multi | 0.70 | bayes_qmix_independent | posterior_sample_shared | 3 | -8.348 | 0.281 | 0.582 | 0.105 |
| multi | 1.00 | bayes_qmix_independent | posterior_sample_shared | 3 | -9.062 | 0.233 | 0.571 | 0.140 |
| multi | 0.70 | bayes_qmix_shared | posterior_mean | 3 | -8.682 | 0.278 | 0.642 | 0.074 |
| multi | 1.00 | bayes_qmix_shared | posterior_mean | 3 | -8.396 | 0.353 | 0.640 | 0.074 |
| multi | 0.70 | bayes_qmix_shared | posterior_sample_independent | 3 | -8.562 | 0.220 | 0.611 | 0.071 |
| multi | 1.00 | bayes_qmix_shared | posterior_sample_independent | 3 | -8.683 | 0.261 | 0.599 | 0.089 |
| multi | 0.70 | bayes_qmix_shared | posterior_sample_shared | 3 | -8.450 | 0.278 | 0.595 | 0.100 |
| multi | 1.00 | bayes_qmix_shared | posterior_sample_shared | 3 | -8.736 | 0.276 | 0.575 | 0.123 |
| multi | 0.70 | qmix_ddqn | deterministic | 3 | -6.552 | 0.484 | 0.564 | 0.098 |
| multi | 1.00 | qmix_ddqn | deterministic | 3 | -7.507 | 0.347 | 0.566 | 0.100 |
| multi | 0.70 | shared_bdqn | posterior_mean | 3 | -6.736 | 0.357 | 0.544 | 0.103 |
| multi | 1.00 | shared_bdqn | posterior_mean | 3 | -7.605 | 0.303 | 0.544 | 0.117 |
| multi | 0.70 | shared_bdqn | posterior_sample_shared | 3 | -7.455 | 0.274 | 0.546 | 0.114 |
| multi | 1.00 | shared_bdqn | posterior_sample_shared | 3 | -8.160 | 0.236 | 0.531 | 0.129 |
| multi | 0.70 | shared_ddqn | deterministic | 3 | -7.441 | 0.307 | 0.570 | 0.094 |
| multi | 1.00 | shared_ddqn | deterministic | 3 | -7.650 | 0.306 | 0.549 | 0.102 |
| single | 0.70 | baseline_frontier | deterministic | 1 | 12.740 | 3.633 | 0.849 | nan |
| single | 1.00 | baseline_frontier | deterministic | 1 | 15.504 | 3.883 | 0.813 | nan |
| single | 0.70 | baseline_oracle | deterministic | 1 | 20.526 | 4.000 | 0.359 | nan |
| single | 1.00 | baseline_oracle | deterministic | 1 | 20.505 | 4.000 | 0.359 | nan |
| single | 0.70 | baseline_random | deterministic | 1 | -3.063 | 0.060 | 0.280 | nan |
| single | 1.00 | baseline_random | deterministic | 1 | -3.124 | 0.060 | 0.280 | nan |
| single | 0.70 | bdqn | posterior_mean | 3 | -2.545 | 0.220 | 0.301 | nan |
| single | 1.00 | bdqn | posterior_mean | 3 | -3.089 | 0.163 | 0.322 | nan |
| single | 0.70 | bdqn | posterior_sample | 3 | -2.642 | 0.154 | 0.271 | nan |
| single | 1.00 | bdqn | posterior_sample | 3 | -2.899 | 0.082 | 0.273 | nan |
| single | 0.70 | ddqn | deterministic | 3 | -4.498 | 0.090 | 0.446 | nan |
| single | 1.00 | ddqn | deterministic | 3 | -5.282 | 0.039 | 0.475 | nan |

## Comparaisons appariées

| pD | Comparaison | Paires | Δ reward | Δ completed | Δ coverage | Δ collision |
|---:|---|---:|---:|---:|---:|---:|
| 0.70 | bayes_qmix_independent_mean_minus_qmix | 3 | -1.534 | -0.172 | 0.070 | -0.031 |
| 1.00 | bayes_qmix_independent_mean_minus_qmix | 3 | -1.092 | -0.034 | 0.084 | -0.032 |
| 0.70 | bayes_qmix_independent_sample_minus_qmix | 3 | -1.502 | -0.189 | 0.039 | -0.023 |
| 1.00 | bayes_qmix_independent_sample_minus_qmix | 3 | -1.350 | -0.118 | 0.032 | -0.005 |
| 0.70 | bayes_qmix_shared_mean_minus_qmix | 3 | -2.131 | -0.207 | 0.078 | -0.023 |
| 1.00 | bayes_qmix_shared_mean_minus_qmix | 3 | -0.889 | 0.007 | 0.074 | -0.026 |
| 0.70 | bayes_qmix_shared_sample_minus_qmix | 3 | -1.898 | -0.207 | 0.031 | 0.002 |
| 1.00 | bayes_qmix_shared_sample_minus_qmix | 3 | -1.230 | -0.071 | 0.009 | 0.023 |
| 0.70 | bdqn_mean_minus_ddqn | 3 | 1.953 | 0.130 | -0.145 | nan |
| 1.00 | bdqn_mean_minus_ddqn | 3 | 2.193 | 0.124 | -0.153 | nan |
| 0.70 | bdqn_sample_minus_ddqn | 3 | 1.856 | 0.064 | -0.174 | nan |
| 1.00 | bdqn_sample_minus_ddqn | 3 | 2.384 | 0.043 | -0.202 | nan |
| 0.70 | independent_sampling_minus_shared_sampling | 3 | 0.397 | 0.018 | 0.008 | -0.025 |
| 1.00 | independent_sampling_minus_shared_sampling | 3 | -0.121 | -0.047 | 0.023 | -0.028 |
| 0.70 | qmix_minus_shared_ddqn | 3 | 0.890 | 0.178 | -0.006 | 0.003 |
| 1.00 | qmix_minus_shared_ddqn | 3 | 0.144 | 0.041 | 0.017 | -0.002 |
| 0.70 | shared_bdqn_mean_minus_shared_ddqn | 3 | 0.705 | 0.050 | -0.026 | 0.009 |
| 1.00 | shared_bdqn_mean_minus_shared_ddqn | 3 | 0.045 | -0.002 | -0.005 | 0.015 |
| 0.70 | shared_bdqn_sample_minus_shared_ddqn | 3 | -0.014 | -0.032 | -0.023 | 0.020 |
| 1.00 | shared_bdqn_sample_minus_shared_ddqn | 3 | -0.510 | -0.070 | -0.018 | 0.028 |
