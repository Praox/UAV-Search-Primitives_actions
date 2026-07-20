# Diagnostic automatique des runs thesis-v2

## État global

- `complete`: 35

## Alertes

- **bayes_qmix_independent seed 43 pD=1.0**: régression tardive: completed final=3.000, max=4.000; fort écart sampled-mean sur completion: -1.200
- **bayes_qmix_independent seed 44 pD=1.0**: régression tardive: completed final=3.000, max=4.000; fort écart sampled-mean sur completion: -2.320
- **bayes_qmix_shared seed 44 pD=1.0**: fort écart sampled-mean sur coverage: -0.222
- **bayes_qmix_shared seed 45 pD=1.0**: fort écart sampled-mean sur coverage: -0.182
- **qmix_ddqn seed 43 pD=1.0**: régression tardive: completed final=2.000, max=4.000; collision finale élevée: 0.302
- **shared_bdqn seed 42 pD=1.0**: régression tardive: completed final=1.000, max=2.000; fort écart sampled-mean sur completion: -0.700
- **shared_bdqn seed 43 pD=1.0**: fort écart sampled-mean sur completion: -2.620
- **shared_bdqn seed 44 pD=1.0**: régression tardive: completed final=2.000, max=3.000; fort écart sampled-mean sur completion: -1.140
- **shared_bdqn seed 45 pD=1.0**: régression tardive: completed final=2.000, max=3.000; collision finale élevée: 0.400; fort écart sampled-mean sur completion: -0.910
- **shared_ddqn seed 42 pD=1.0**: collision finale élevée: 0.347
- **shared_ddqn seed 43 pD=1.0**: collision finale élevée: 0.448
- **shared_ddqn seed 46 pD=1.0**: régression tardive: completed final=2.000, max=3.000
- **bdqn seed 43 pD=1.0**: régression tardive: completed final=1.000, max=2.000
- **bdqn seed 44 pD=1.0**: régression tardive: completed final=0.000, max=2.000; fort écart sampled-mean sur completion: -1.160
- **bdqn seed 45 pD=1.0**: régression tardive: completed final=1.000, max=2.000
- **ddqn seed 42 pD=1.0**: régression tardive: completed final=1.000, max=2.000
- **ddqn seed 43 pD=1.0**: régression tardive: completed final=2.000, max=3.000

## Interprétation

Une alerte n'implique pas automatiquement qu'un run est invalide. Elle indique un point à vérifier dans les courbes, le log d'entraînement et l'évaluation appariée.
