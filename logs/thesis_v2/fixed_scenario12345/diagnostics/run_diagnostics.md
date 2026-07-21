# Diagnostic automatique des runs thesis-v2

## État global

- `complete`: 35

## Alertes

- **shared_bdqn seed 42 pD=1.0**: régression tardive: completed final=3.000, max=4.000; fort écart sampled-mean sur completion: -1.200; fort écart sampled-mean sur coverage: +0.277
- **shared_bdqn seed 43 pD=1.0**: fort écart sampled-mean sur completion: -0.900; fort écart sampled-mean sur coverage: +0.229
- **shared_bdqn seed 44 pD=1.0**: régression tardive: completed final=2.000, max=3.000; fort écart sampled-mean sur coverage: -0.163
- **shared_bdqn seed 46 pD=1.0**: fort écart sampled-mean sur coverage: +0.184
- **bdqn seed 42 pD=1.0**: régression tardive: completed final=2.000, max=3.000; fort écart sampled-mean sur completion: -0.890; fort écart sampled-mean sur coverage: -0.153

## Interprétation

Une alerte n'implique pas automatiquement qu'un run est invalide. Elle indique un point à vérifier dans les courbes, le log d'entraînement et l'évaluation appariée.
