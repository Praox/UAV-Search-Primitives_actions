# Diagnostic automatique des runs thesis-v2

## État global

- `complete`: 39
- `trained`: 1

## Alertes

- **qmix_ddqn seed 44 pD=0.5**: collision finale élevée: 0.208
- **qmix_ddqn seed 44 pD=0.7**: régression tardive: completed final=0.255, max=0.530; collision finale élevée: 0.254
- **bayes_qmix_independent seed 46 pD=1.0**: évaluation détaillée finale absente
- **qmix_ddqn seed 43 pD=1.0**: régression tardive: completed final=0.320, max=0.590; collision finale élevée: 0.233
- **qmix_ddqn seed 45 pD=1.0**: régression tardive: completed final=0.190, max=0.540

## Interprétation

Une alerte n'implique pas automatiquement qu'un run est invalide. Elle indique un point à vérifier dans les courbes, le log d'entraînement et l'évaluation appariée.
