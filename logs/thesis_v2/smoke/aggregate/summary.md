# Résumé agrégé de l'étude thesis-v2

Les intervalles des méthodes apprises sont des IC Student-t à 95 % sur les seeds d'entraînement.
Les intervalles des baselines sont calculés sur les mondes d'évaluation.

## Méthodes

| Scope | pD | Méthode | Politique | Seeds | Reward | Completed | Coverage | Collision |
|---|---:|---|---|---:|---:|---:|---:|---:|
| multi | 1.00 | baseline_local_frontier | deterministic | 1 | 17.015 | 4.000 | 0.752 | 0.262 |
| multi | 1.00 | baseline_random | deterministic | 1 | -6.828 | 0.000 | 0.502 | 0.008 |
| multi | 1.00 | qmix_ddqn | deterministic | 1 | -6.496 | 0.250 | 0.519 | 0.129 |
| multi | 1.00 | shared_ddqn | deterministic | 1 | -4.850 | 0.750 | 0.394 | 0.179 |
| single | 1.00 | baseline_frontier | deterministic | 1 | 12.265 | 3.750 | 0.929 | nan |
| single | 1.00 | baseline_oracle | deterministic | 1 | 20.696 | 4.000 | 0.336 | nan |
| single | 1.00 | baseline_random | deterministic | 1 | -3.318 | 0.000 | 0.249 | nan |
| single | 1.00 | ddqn | deterministic | 1 | -3.534 | 0.000 | 0.214 | nan |

## Comparaisons appariées

| pD | Comparaison | Paires | Δ reward | Δ completed | Δ coverage | Δ collision |
|---:|---|---:|---:|---:|---:|---:|
| 1.00 | qmix_minus_shared_ddqn | 1 | -1.646 | -0.500 | 0.126 | -0.050 |
