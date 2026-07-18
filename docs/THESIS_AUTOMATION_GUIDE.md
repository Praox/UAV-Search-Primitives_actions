# Automatisation de l'étude thesis-v2

Cette extension doit être appliquée **après** le correctif `uav_thesis_correction_overlay`.
Elle ajoute :

- une évaluation détaillée et reproductible des checkpoints single-UAV ;
- une évaluation détaillée des checkpoints multi-UAV ;
- les baselines corrigées dans les mêmes environnements ;
- un runner qui automatise seeds, probabilités, warm-start BDQN, évaluations et reprises ;
- l'agrégation Student-t, les comparaisons appariées et la différence-de-différences ;
- les courbes d'apprentissage et les graphiques finaux ;
- un diagnostic automatique des runs et des logs.

---

## 1. Installation

Depuis la racine du dépôt, après avoir installé le premier overlay :

```bash
/path/to/uav_thesis_automation_overlay/apply_automation_overlay.sh .
pip install -e .
pip install -r requirements-thesis.txt
pytest -q tests/test_thesis_corrections.py tests/test_thesis_automation.py
```

Le script sauvegarde les versions précédentes de `train_thesis_single.py` et
`train_thesis_multi.py` avant de les remplacer.

---

## 2. Structure des sorties

Le runner crée deux arbres séparés.

### Checkpoints et métriques d'entraînement

```text
runs/thesis_v2/<stage>/<pdet>/<scope>/<algo>/seed<seed>/
├── run_config.json
├── metrics.csv
├── best.pt
├── latest.pt
├── training_status.json
└── evaluation/
    ├── final_test_<mode>_episodes.csv
    ├── final_test_<mode>_summary.json
    └── final_test_evaluation_index.json
```

### Logs, baselines, agrégats et figures

```text
logs/thesis_v2/<stage>/
├── manifest.jsonl
├── pdet_1p00/
│   ├── single/<algo>/seed42_train.log
│   └── multi/<algo>/seed42_train.log
├── baselines/
├── aggregate/
│   ├── all_runs.csv
│   ├── summary_by_method.csv
│   ├── paired_by_seed.csv
│   ├── paired_summary.csv
│   ├── difference_in_differences.csv
│   ├── learning_curves.csv
│   └── summary.md
├── plots/
└── diagnostics/
    ├── run_diagnostics.csv
    └── run_diagnostics.md
```

`metrics.csv` contient les points périodiques utilisés pour les courbes. Les CSV
`*_episodes.csv` contiennent un point par monde d'évaluation et permettent des
analyses appariées plus fines.

---

## 3. Vérifier le plan avant de lancer

Le dry-run affiche toutes les commandes sans entraîner :

```bash
python scripts/run_thesis_suite.py screen --dry-run
```

À vérifier :

- les seeds attendues ;
- `pD=1.0` et `pD=0.7` ;
- les chemins de warm-start BDQN ;
- le nombre d'épisodes ;
- `reward_mode=task_potential` ;
- `global_state_mode=memory_union` pour l'étude principale.

---

## 4. Étape 0 — smoke test

```bash
python scripts/run_thesis_suite.py smoke --device cpu
```

Le smoke test exécute :

1. les tests unitaires ;
2. un petit DDQN single ;
3. un petit shared-DDQN ;
4. un petit QMIX-DDQN ;
5. quelques mondes de baseline ;
6. l'agrégation, les figures et le diagnostic.

### Résultat attendu

L'objectif n'est pas la performance. Il faut seulement obtenir :

- aucun traceback ;
- `best.pt` et `latest.pt` présents ;
- au moins deux lignes dans chaque `metrics.csv` ;
- des JSON d'évaluation valides ;
- des fichiers dans `aggregate/`, `plots/` et `diagnostics/`.

### Points à surveiller

- une `loss` finie après le début de l'apprentissage ;
- aucune action masquée sélectionnée ;
- aucun problème de forme des tenseurs ;
- pour QMIX, `q_tot_mean` et `target_mean` finis ;
- absence de duplication `epsilon_start` dans Bayesian-QMIX.

Ne tire aucune conclusion scientifique du smoke test.

---

## 5. Étape 1 — screening complet

```bash
python scripts/run_thesis_suite.py screen --device cuda
```

Valeurs par défaut :

- seeds `42,43,44` ;
- probabilités `1.0,0.7` ;
- 400 épisodes d'entraînement ;
- 100 mondes de validation par checkpoint ;
- 300 mondes de test final ;
- single : DDQN puis BDQN warm-starté ;
- multi : shared-DDQN, shared-BDQN diagnostic, QMIX-DDQN,
  Bayesian-QMIX shared et independent ;
- random/frontier/oracle single ;
- random/local-frontier multi.

Le runner est reprenable. Une seconde exécution ignore les jobs qui possèdent déjà
leur index d'évaluation finale. Utiliser `--force` uniquement pour refaire volontairement
les résultats.

### Exécuter seulement une partie

Single-UAV :

```bash
python scripts/run_thesis_suite.py screen \
  --scope single \
  --device cuda
```

Multi-UAV :

```bash
python scripts/run_thesis_suite.py screen \
  --scope multi \
  --device cuda
```

Algorithmes contrôlés :

```bash
python scripts/run_thesis_suite.py custom \
  --scope multi \
  --seeds 42-44 \
  --probabilities 1.0,0.7 \
  --multi-algos shared_ddqn,qmix_ddqn \
  --train-episodes 400 \
  --final-test-episodes 300 \
  --device cuda
```

---

## 6. Que regarder dans les courbes d'apprentissage

Les courbes sont générées dans :

```text
logs/thesis_v2/screen/plots/learning_curves/
```

Chaque figure montre :

- les trajectoires fines de chaque seed ;
- la moyenne épaisse ;
- l'IC Student-t 95 % sur les seeds lorsque plusieurs seeds sont disponibles.

### 6.1 Reward

Le reward est utile pour diagnostiquer l'optimisation, mais il n'est pas le critère
principal. Surveiller :

- une croissance moyenne ;
- l'absence d'explosion ;
- l'absence d'effondrement tardif ;
- la cohérence avec la completion.

Une reward qui augmente alors que la completion baisse indique encore un problème
d'alignement ou de shaping.

### 6.2 Completed et completed value

Ce sont les métriques principales. Chercher :

- une amélioration par rapport à random ;
- une progression au cours de l'entraînement ;
- une variance inter-seed acceptable ;
- une amélioration de `completed_value` et pas seulement du nombre de petites cibles.

### 6.3 Coverage

La couverture mesure le comportement de recherche. Une couverture élevée n'est positive
que si elle se convertit ensuite en détection et completion.

Cas problématique :

```text
coverage ↑, completed ↔ ou ↓
```

Cela signifie que l'agent apprend surtout à explorer et pas à s'engager dans le tracking.

### 6.4 STAY et tracking-progress ratio

Avec le tracking intentionnel, `STAY` doit apparaître lorsqu'une cible est connue et à
portée. Surveiller simultanément :

- `stay_ratio` ;
- `tracking_progress_ratio` ;
- `completed_mean`.

Un STAY élevé avec peu de progression indique une politique bloquée. Un STAY presque nul
avec peu de completion indique que l'agent refuse de maintenir le tracking.

### 6.5 Collision et overlap multi-UAV

Une amélioration QMIX crédible doit idéalement produire :

- completion supérieure ;
- collisions inférieures ;
- overlap inférieur ou contrôlé ;
- couverture égale ou supérieure.

Une amélioration de reward expliquée uniquement par une baisse des collisions doit être
rapportée comme un gain de déconfliction, pas comme un gain de recherche complet.

### 6.6 Loss, Q et targets

Surveiller :

- `loss` finie ;
- `q_mean/q_tot_mean` du même ordre de grandeur que `target_mean` ;
- aucune dérive continue des Q-values ;
- aucune oscillation périodique très forte.

Huber et les soft targets doivent réduire les pics, pas nécessairement rendre la courbe
parfaitement monotone.

---

## 7. Points Bayesian à ne pas manquer

### 7.1 BDQN

Dans `metrics.csv` et le diagnostic :

- `posterior_rebuilds >= 1` ;
- `td_residual_variance` finie ;
- features gelées pour l'expérience principale ;
- comparaison séparée `posterior_mean` / `posterior_sample`.

Si aucun rebuild n'a lieu, le résultat BDQN n'est pas interprétable.

### 7.2 Bayesian-QMIX

Surveiller :

- `posterior_std_mean` ;
- `posterior_std_min` et `posterior_std_max` ;
- `posterior_kl_per_parameter` ;
- `td_loss` et `kl_loss` séparément.

Signatures problématiques :

- std proche de `1e-4` : posterior collapse, comportement presque déterministe ;
- std proche de `1.0` : incertitude non contrôlée ;
- KL dominant largement le TD loss : régularisation trop forte ;
- sampled beaucoup meilleur que mean sur une seule seed : possible tirage chanceux ;
- sampled beaucoup plus mauvais que mean : posterior mal calibré.

Les évaluations Bayesian-QMIX produisent automatiquement :

- `posterior_mean` ;
- `posterior_sample_shared` ;
- `posterior_sample_independent`.

Cela permet de séparer l'effet de l'entraînement du mode d'exécution.

---

## 8. Évaluer manuellement un checkpoint

### Single-UAV

```bash
python scripts/evaluate_thesis_single.py \
  --run-dir runs/thesis_v2/screen/pdet_0p70/single/bdqn/seed42 \
  --checkpoint best \
  --episodes 1000 \
  --eval-seed-base 200000 \
  --policy-modes posterior_mean,posterior_sample \
  --device cuda
```

### Multi-UAV

```bash
python scripts/evaluate_thesis_multi.py \
  --run-dir runs/thesis_v2/screen/pdet_0p70/multi/bayes_qmix_shared/seed42 \
  --checkpoint best \
  --episodes 1000 \
  --policy-modes posterior_mean,posterior_sample_shared,posterior_sample_independent \
  --device cuda
```

Les mêmes mondes sont utilisés pour tous les modes lorsque `eval-seed-base` est identique.

---

## 9. Évaluer les baselines seules

```bash
python scripts/evaluate_thesis_baselines.py \
  --scope both \
  --probabilities 1.0,0.7 \
  --episodes 1000 \
  --eval-seed-base 200000
```

Les baselines single sont :

- random ;
- frontier ;
- oracle.

Les baselines multi sont :

- random ;
- local-frontier.

Le local-frontier est la baseline opérationnelle la plus importante : il indique si la
complexité MARL apporte plus qu'une heuristique décentralisée structurée.

---

## 10. Agréger ou refaire les figures sans réentraîner

```bash
python scripts/run_thesis_suite.py aggregate --stage screen
```

Ou séparément :

```bash
python scripts/aggregate_thesis_results.py \
  --run-root runs/thesis_v2/screen \
  --baseline-root logs/thesis_v2/screen/baselines \
  --output-dir logs/thesis_v2/screen/aggregate

python scripts/plot_thesis_results.py \
  --aggregate-dir logs/thesis_v2/screen/aggregate \
  --output-dir logs/thesis_v2/screen/plots
```

---

## 11. Lire automatiquement l'état des runs

```bash
python scripts/run_thesis_suite.py status --stage screen
```

Le diagnostic signale notamment :

- checkpoint ou évaluation manquante ;
- traceback dans les logs ;
- loss finale non finie ;
- absence de posterior rebuild ;
- posterior proche de ses bornes ;
- collision ou overlap élevés ;
- régression tardive de completion ;
- différence sampled/mean anormalement grande.

Lire ensuite :

```text
logs/thesis_v2/screen/diagnostics/run_diagnostics.md
```

---

## 12. Interpréter les fichiers agrégés

### `summary_by_method.csv`

Moyenne et IC de chaque méthode. Pour une méthode apprise, l'unité statistique est la
seed d'entraînement, pas les 300 ou 1000 mondes internes.

### `paired_by_seed.csv`

Différence gauche-droite pour chaque seed appariée.

Exemple :

```text
qmix_minus_shared_ddqn
```

isole l'effet observé après correction des budgets et des masques.

### `paired_summary.csv`

Moyenne et IC Student-t des différences appariées. Pour une métrique où « plus est mieux »,
un IC entièrement positif est une preuve plus forte qu'une simple différence de moyennes.
Pour collision/overlap, une différence négative est généralement favorable.

### `difference_in_differences.csv`

Calcule :

```text
(Bayesian - déterministe à pD=0.7)
-
(Bayesian - déterministe à pD=1.0)
```

Sur completion, une valeur positive soutient l'hypothèse que l'incertitude bayésienne
devient relativement plus utile quand le capteur est bruité.

---

## 13. Gates de décision après le screening

### Gate A — environnement et baseline déterministe

Continuer seulement si :

- DDQN bat random sur completion ou completed value ;
- la frontier reste une borne structurée forte ;
- les courbes ne présentent pas de NaN ;
- le tracking intentionnel produit un ratio de progression non nul.

Sinon, corriger l'environnement ou la reward avant d'étudier Bayesian-QMIX.

### Gate B — crédit centralisé

Le résultat QMIX est intéressant si :

- `QMIX - shared-DDQN` améliore completion/completed value ;
- ou réduit clairement collisions/overlap sans dégrader fortement completion ;
- l'effet est cohérent sur plusieurs seeds.

### Gate C — effet bayésien

Conserver pour confirmation le meilleur mode Bayesian-QMIX si :

- son posterior ne s'effondre pas ;
- il améliore au moins une métrique mission critique ;
- le gain ne provient pas d'une seule seed ;
- le gain est plus visible ou plus stable à `pD=0.7`.

### Gate D — confirmation

Exemple :

```bash
python scripts/run_thesis_suite.py confirm \
  --multi-algos shared_ddqn,qmix_ddqn,bayes_qmix_shared \
  --device cuda
```

La confirmation utilise par défaut :

- seeds 42-48 ;
- 1000 épisodes d'entraînement ;
- 1000 mondes de test ;
- uniquement `pD=0.7`.

Ne retune pas les hyperparamètres après avoir vu les résultats de confirmation.

---

## 14. Commandes utiles de reprise

Refaire uniquement les évaluations finales :

```bash
python scripts/run_thesis_suite.py evaluate --stage screen --device cuda
```

Refaire les figures :

```bash
python scripts/run_thesis_suite.py aggregate --stage screen
```

Ajouter des seeds sans relancer les anciennes :

```bash
python scripts/run_thesis_suite.py custom \
  --stage screen \
  --seeds 45-48 \
  --probabilities 1.0,0.7 \
  --device cuda
```

Le runner ignore les jobs déjà complets sauf avec `--force`.
