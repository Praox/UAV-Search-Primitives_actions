# Résumé agrégé de l'étude thesis-v2

Les intervalles des méthodes apprises sont des IC Student-t à 95 % sur les seeds d'entraînement.
Les intervalles des baselines sont calculés sur les mondes d'évaluation.

## Méthodes

| Scope | pD | Méthode | Politique | Seeds | Reward | Completed | Coverage | Collision |
|---|---:|---|---|---:|---:|---:|---:|---:|
| multi | 1.00 | baseline_local_frontier | deterministic | 1 | 14.207 | 3.720 | 0.728 | 0.378 |
| multi | 1.00 | baseline_random | deterministic | 1 | -8.502 | 0.260 | 0.636 | 0.003 |
| multi | 1.00 | bayes_qmix_independent | posterior_mean | 5 | 22.639 | 4.000 | 0.369 | 0.000 |
| multi | 1.00 | bayes_qmix_independent | posterior_sample_independent | 5 | 22.229 | 3.994 | 0.350 | 0.004 |
| multi | 1.00 | bayes_qmix_independent | posterior_sample_shared | 5 | 22.414 | 3.996 | 0.349 | 0.006 |
| multi | 1.00 | bayes_qmix_shared | posterior_mean | 5 | 22.835 | 4.000 | 0.325 | 0.000 |
| multi | 1.00 | bayes_qmix_shared | posterior_sample_independent | 5 | 22.098 | 4.000 | 0.326 | 0.001 |
| multi | 1.00 | bayes_qmix_shared | posterior_sample_shared | 5 | 22.083 | 3.996 | 0.325 | 0.001 |
| multi | 1.00 | qmix_ddqn | deterministic | 5 | 21.999 | 4.000 | 0.447 | 0.000 |
| multi | 1.00 | shared_bdqn | posterior_mean | 5 | 18.027 | 3.800 | 0.493 | 0.000 |
| multi | 1.00 | shared_bdqn | posterior_sample_shared | 5 | 7.512 | 3.186 | 0.593 | 0.079 |
| multi | 1.00 | shared_ddqn | deterministic | 5 | 22.266 | 4.000 | 0.358 | 0.000 |
| single | 1.00 | baseline_frontier | deterministic | 1 | 13.866 | 3.710 | 0.830 | nan |
| single | 1.00 | baseline_oracle | deterministic | 1 | 21.047 | 4.000 | 0.289 | nan |
| single | 1.00 | baseline_random | deterministic | 1 | -3.273 | 0.100 | 0.276 | nan |
| single | 1.00 | bdqn | posterior_mean | 5 | 18.484 | 3.800 | 0.354 | nan |
| single | 1.00 | bdqn | posterior_sample | 5 | 15.048 | 3.434 | 0.338 | nan |
| single | 1.00 | ddqn | deterministic | 5 | 21.064 | 4.000 | 0.295 | nan |

## Comparaisons appariées

| pD | Comparaison | Paires | Δ reward | Δ completed | Δ coverage | Δ collision |
|---:|---|---:|---:|---:|---:|---:|
| 1.00 | bayes_qmix_independent_mean_minus_qmix | 5 | 0.639 | 0.000 | -0.078 | 0.000 |
| 1.00 | bayes_qmix_independent_sample_minus_qmix | 5 | 0.230 | -0.006 | -0.096 | 0.004 |
| 1.00 | bayes_qmix_shared_mean_minus_qmix | 5 | 0.836 | 0.000 | -0.122 | 0.000 |
| 1.00 | bayes_qmix_shared_sample_minus_qmix | 5 | 0.084 | -0.004 | -0.121 | 0.001 |
| 1.00 | bdqn_mean_minus_ddqn | 5 | -2.581 | -0.200 | 0.059 | nan |
| 1.00 | bdqn_sample_minus_ddqn | 5 | -6.016 | -0.566 | 0.042 | nan |
| 1.00 | independent_sampling_minus_shared_sampling | 5 | 0.146 | -0.002 | 0.025 | 0.003 |
| 1.00 | qmix_minus_shared_ddqn | 5 | -0.267 | 0.000 | 0.088 | 0.000 |
| 1.00 | shared_bdqn_mean_minus_shared_ddqn | 5 | -4.239 | -0.200 | 0.134 | 0.000 |
| 1.00 | shared_bdqn_sample_minus_shared_ddqn | 5 | -14.754 | -0.814 | 0.235 | 0.079 |
