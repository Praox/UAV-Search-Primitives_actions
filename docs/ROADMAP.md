# Roadmap accélérée mémoire — UAV search/track RL

## Objectif expérimental

Comparer progressivement :

1. `DQN` single drone : actions primitives, belief map.
2. `DDQN` single drone : même setup, target Double DQN.
3. `BDQN` single drone : même setup, meilleure exploration via posterior sampling sur la tête Q.
4. `3 drones shared BDQN independent` : parameter sharing, pas de mixer.
5. `QMIX-DDQN` : vraie coordination MARL CTDE.
6. `QMIX-BDQN` : uniquement si le temps reste.

## Pourquoi un nouveau repo plutôt que modifier brutalement l'ancien

Le repo de base `Praox/bdqn-uav-search-n-track` apprend une décision haut niveau `SEARCH/TRACK`, puis un `MissionController` transforme cette mission en mouvement. Ici, le but du mémoire est différent : l'agent doit directement choisir les mouvements `stay/up/down/left/right`.

Donc ce repo garde la structure mentale de l'ancien, mais change le point où l'intelligence est apprise :

```text
Avant : obs -> BDQN -> SEARCH/TRACK -> MissionController -> mouvement
Ici  : obs -> DQN/DDQN/BDQN -> mouvement direct
```

## Étape 1 — DQN / DDQN single drone

Definition of done :

- `python scripts/smoke_test.py` passe.
- `python scripts/train.py --algo dqn --episodes 500 --run-dir runs/dqn` produit `best.pt` et `latest.pt`.
- `python scripts/train.py --algo ddqn --episodes 500 --run-dir runs/ddqn` produit `best.pt` et `latest.pt`.
- `scripts/evaluate.py` donne les métriques : reward, detected, completed, coverage, stay ratio, revisit ratio, boundary ratio.

Ce que tu compares :

- DQN vs DDQN sur les mêmes seeds.
- Reward moyenne.
- Nombre de cibles détectées.
- Nombre de cibles complétées.
- Couverture de la carte.
- Comportement : trop de stay ? trop de revisits ? trop de boundaries ?

## Étape 2 — BDQN single drone

Definition of done :

- `python scripts/train.py --algo bdqn --episodes 500 --run-dir runs/bdqn` produit `best.pt` et `latest.pt`.
- Même environnement, mêmes rewards, mêmes seeds que DQN/DDQN.
- Aucune différence de réseau sauf la tête bayésienne.

Hypothèse mémoire :

BDQN doit mieux explorer que DQN/DDQN, surtout au début, parce que le posterior sampling produit une politique plus cohérente sur l'épisode que l'epsilon-greedy pas-à-pas.

## Étape 3 — 3 drones shared BDQN independent

Definition of done :

- `python scripts/train_shared_bdqn.py --episodes 500 --n-agents 3 --run-dir runs/shared_bdqn_3uav` tourne sans crash.
- La couverture d'équipe doit dépasser le single drone.
- La completion doit être au moins comparable ou meilleure.

Ce que cette étape n'est pas :

- Ce n'est pas QMIX.
- Ce n'est pas une vraie factorisation de valeur d'équipe.
- C'est une baseline rapide : même politique partagée, reward d'équipe, mémoire d'équipe.

## Étape 4 — QMIX-DDQN

À faire ensuite :

- Ajouter un replay buffer épisodique.
- Stocker `obs[n_agents]`, `actions[n_agents]`, `reward_team`, `next_obs[n_agents]`, `state_global`, `next_state_global`, `done`.
- Utiliser `QNetwork` partagé pour obtenir `Q_i`.
- Utiliser `QMixer` pour obtenir `Q_tot`.
- Appliquer target Double DQN : action selection via online utilities, evaluation via target utilities + target mixer.

Le fichier `src/uav_search_belief20/marl/qmix_mixer.py` est déjà prêt comme brique de base.

## Étape 5 — QMIX-BDQN

Stretch goal.

Version minimale possible : remplacer le réseau utilité DDQN par une tête bayésienne ou bootstrapped par agent. Ne pas y passer trop de temps si QMIX-DDQN n'est pas déjà stable.

## Expériences minimales pour le mémoire

Pour chaque méthode single-drone :

```text
seeds = 5
episodes = 500 ou 1000 si temps OK
évaluation = 200 ou 500 épisodes
```

Méthodes single :

```text
DQN
DDQN
BDQN
```

Méthodes multi :

```text
Shared BDQN independent, 3 UAV
QMIX-DDQN, 3 UAV si implémenté à temps
```

Figures à produire :

- reward moyenne par épisode ;
- completed_mean par évaluation ;
- detected_mean par évaluation ;
- sensor_coverage_ratio_mean ;
- action ratios, surtout `stay_ratio`, `revisit_ratio`, `boundary_hit_ratio`.

## Ordre de travail recommandé

1. Ne touche plus aux rewards tant que DQN/DDQN/BDQN single ne tournent pas.
2. Stabilise l'environnement primitive-action.
3. Fais DQN puis DDQN.
4. Fais BDQN.
5. Lance toutes les seeds single-drone.
6. Pendant que ça tourne, avance le multi-drone shared BDQN.
7. Seulement ensuite, QMIX-DDQN.
