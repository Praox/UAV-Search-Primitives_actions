# Résumé agrégé de l'étude thesis-v2

Les intervalles des méthodes apprises sont des IC Student-t à 95 % sur les seeds d'entraînement.
Les intervalles des baselines sont calculés sur les mondes d'évaluation.

## Méthodes

| Scope | pD | Méthode | Politique | Seeds | Reward | Completed | Coverage | Collision |
|---|---:|---|---|---:|---:|---:|---:|---:|
| multi | 1.00 | baseline_local_frontier | deterministic | 1 | 14.207 | 3.720 | 0.728 | 0.378 |
| multi | 1.00 | baseline_random | deterministic | 1 | -8.502 | 0.260 | 0.636 | 0.003 |
| multi | 1.00 | bayes_qmix_independent | posterior_mean | 5 | 18.778 | 4.000 | 0.470 | 0.000 |
| multi | 1.00 | bayes_qmix_independent | posterior_sample_independent | 5 | 11.285 | 3.270 | 0.478 | 0.042 |
| multi | 1.00 | bayes_qmix_independent | posterior_sample_shared | 5 | 11.370 | 3.266 | 0.474 | 0.049 |
| multi | 1.00 | bayes_qmix_shared | posterior_mean | 5 | 9.513 | 3.400 | 0.562 | 0.000 |
| multi | 1.00 | bayes_qmix_shared | posterior_sample_independent | 5 | 10.332 | 3.464 | 0.496 | 0.031 |
| multi | 1.00 | bayes_qmix_shared | posterior_sample_shared | 5 | 10.208 | 3.460 | 0.487 | 0.039 |
| multi | 1.00 | qmix_ddqn | deterministic | 5 | 13.826 | 3.600 | 0.491 | 0.000 |
| multi | 1.00 | shared_bdqn | posterior_mean | 5 | 0.469 | 2.800 | 0.592 | 0.082 |
| multi | 1.00 | shared_bdqn | posterior_sample_shared | 5 | -5.983 | 1.644 | 0.599 | 0.082 |
| multi | 1.00 | shared_ddqn | deterministic | 5 | 8.961 | 3.200 | 0.563 | 0.001 |
| single | 1.00 | baseline_frontier | deterministic | 1 | 13.866 | 3.710 | 0.830 | nan |
| single | 1.00 | baseline_oracle | deterministic | 1 | 21.047 | 4.000 | 0.289 | nan |
| single | 1.00 | baseline_random | deterministic | 1 | -3.273 | 0.100 | 0.276 | nan |
| single | 1.00 | bdqn | posterior_mean | 5 | 6.395 | 2.000 | 0.301 | nan |
| single | 1.00 | bdqn | posterior_sample | 5 | 4.850 | 1.752 | 0.305 | nan |
| single | 1.00 | ddqn | deterministic | 5 | 3.561 | 2.000 | 0.411 | nan |

## Comparaisons appariées

| pD | Comparaison | Paires | Δ reward | Δ completed | Δ coverage | Δ collision |
|---:|---|---:|---:|---:|---:|---:|
| 1.00 | bayes_qmix_independent_mean_minus_qmix | 5 | 4.952 | 0.400 | -0.020 | 0.000 |
| 1.00 | bayes_qmix_independent_sample_minus_qmix | 5 | -2.541 | -0.330 | -0.013 | 0.042 |
| 1.00 | bayes_qmix_shared_mean_minus_qmix | 5 | -4.313 | -0.200 | 0.072 | 0.000 |
| 1.00 | bayes_qmix_shared_sample_minus_qmix | 5 | -3.618 | -0.140 | -0.004 | 0.039 |
| 1.00 | bdqn_mean_minus_ddqn | 5 | 2.834 | 0.000 | -0.109 | nan |
| 1.00 | bdqn_sample_minus_ddqn | 5 | 1.289 | -0.248 | -0.106 | nan |
| 1.00 | independent_sampling_minus_shared_sampling | 5 | 1.077 | -0.190 | -0.010 | 0.002 |
| 1.00 | qmix_minus_shared_ddqn | 5 | 4.865 | 0.400 | -0.072 | -0.001 |
| 1.00 | shared_bdqn_mean_minus_shared_ddqn | 5 | -8.492 | -0.400 | 0.029 | 0.082 |
| 1.00 | shared_bdqn_sample_minus_shared_ddqn | 5 | -14.944 | -1.556 | 0.036 | 0.081 |
