# Canonical thesis correction overlay

This overlay is designed for the current `Praox/UAV-Search-Primitives_actions` layout.
It leaves the historical scripts in place for reproduction, but the thesis experiments
should use only:

- `scripts/train_thesis_single.py`
- `scripts/train_thesis_multi.py`
- `ThesisPrimitiveSearchEnv`
- `ThesisMultiDroneLocalMemoryEnv`

The corrected neural architecture is intentionally incompatible with old checkpoints.
Retraining is required. Do not mix old and new curves in one statistical comparison.

---

# 1. Valid-action Bellman backups

## Old problem

The behavior policy masked actions that leave the grid, but the Bellman target maximized
over every action:

\[
a^*=\arg\max_{a\in\mathcal A} Q_\theta(o',a).
\]

An invalid boundary action could therefore have a large, uncorrected Q-value and enter
the bootstrap target even though it could never be executed.

## Correction

Every replay transition stores the masks

\[
m_t,\quad m_{t+n}.
\]

The corrected Double-DQN target is

\[
a^*=\arg\max_{a:m_{t+n}(a)=1}Q_\theta(o_{t+n},a),
\]

\[
y_t=R_t^{(n)}+\gamma^n(1-d_t)
Q_{\bar\theta}(o_{t+n},a^*).
\]

This makes DDQN, BDQN and QMIX optimize the same constrained action process that is
used during execution. It removes a major confound from the QMIX-versus-shared-policy
comparison.

---

# 2. Comparable exploration clocks

## Old problem

The shared DDQN called `act()` once for every UAV, and each call incremented the epsilon
counter. With three UAVs,

\[
\text{clock increment per environment step}=3,
\]

whereas QMIX incremented its counter once. Shared DDQN therefore reached its minimum
epsilon three times earlier in environment time.

## Correction

Shared-policy action selection uses

```python
agent.act(..., advance_env_step=False)
```

for each UAV, followed by

```python
agent.advance_env_step()
```

once for the complete joint transition. Thus

\[
\epsilon_t=\epsilon_{\mathrm{start}}+
\min\left(1,\frac{t_{\mathrm{joint}}}{T_\epsilon}\right)
(\epsilon_{\mathrm{end}}-\epsilon_{\mathrm{start}})
\]

uses the same time variable for shared DDQN and QMIX.

---

# 3. Comparable optimization budgets

A QMIX minibatch with `B` joint transitions contains `B*N` local observations. A shared
learner minibatch previously contained only `B` local observations.

The canonical multi-UAV script uses

\[
B_{\mathrm{shared}}=N_{\mathrm{agents}}B_{\mathrm{QMIX}}.
\]

For three agents and a joint batch of 64, the shared batch is 192. Both updates then
process approximately the same number of local observations. The number of optimizer
steps remains equal.

This is essential for the thesis claim that

\[
J_{\mathrm{QMIX}}-J_{\mathrm{shared}}
\]

measures centralized credit assignment rather than a larger effective minibatch.

---

# 4. Validation worlds and final test worlds

The new scripts use disjoint seed ranges:

- training worlds: generated from the training seed and its RNG stream;
- validation/checkpoint worlds: base seed `100000`;
- final thesis test worlds: base seed `200000`.

The best checkpoint is selected only on validation worlds. The final test worlds are
opened once after all model and hyperparameter decisions have been frozen.

This prevents selecting the checkpoint and reporting performance on the same random
worlds.

---

# 5. Huber TD loss

The old squared TD loss was

\[
\mathcal L_{\mathrm{MSE}}=\frac1B\sum_i e_i^2,
\qquad e_i=Q_i-y_i.
\]

Its gradient is proportional to `2e`, so a rare completion reward can dominate a
minibatch.

The corrected agents use the Huber loss

\[
L_\delta(e)=
\begin{cases}
\frac12e^2,&|e|\le\delta,\\
\delta(|e|-\frac12\delta),&|e|>\delta.
\end{cases}
\]

For large errors the gradient is bounded by `delta`. This does not remove the signal
from valuable completions, but it prevents one transition from causing an excessive
network update. The default is `delta=1`.

---

# 6. Soft target updates

The old target network jumped every 500 optimizer steps. The new default performs
Polyak averaging after every update:

\[
\bar\theta\leftarrow(1-\tau)\bar\theta+\tau\theta,
\qquad \tau=0.005.
\]

The target changes smoothly, which reduces discontinuities in the TD labels. Setting
`--target-tau 0` recovers periodic hard copies for an ablation.

---

# 7. Three-step returns

One-step TD propagates a completion reward backward only one transition per fitted
update. With a horizon of 150, this is slow.

The new replay buffers construct

\[
R_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k},
\]

and

\[
y_t^{(n)}=R_t^{(n)}+\gamma^n(1-d_t)Q_{\mathrm{target}}(o_{t+n},a^*).
\]

The default is `n=3`. Near termination, the buffer automatically uses the shorter
remaining horizon. Multi-agent shared replay maintains one n-step queue per UAV, so
transitions from different agents are never mixed temporally.

A five-step return is a later ablation, not the default. Three steps is safer because
it improves reward propagation without making targets excessively off-policy.

---

# 8. Compact spatial network and normalized features

The previous network flattened `64 x 20 x 20 = 25600` activations directly into a
fully connected layer, producing roughly 6.6 million parameters.

The corrected extractor uses

\[
20\times20\rightarrow10\times10\rightarrow5\times5
\rightarrow\operatorname{AdaptivePool}(4\times4)
\rightarrow256\rightarrow128,
\]

with approximately half a million parameters. This lowers sample complexity and keeps
spatial processing inside convolutional layers for longer.

The output is normalized:

\[
\hat\phi(o)=\frac{\phi(o)}{\max(\|\phi(o)\|_2,10^{-6})}.
\]

For the Bayesian head,

\[
\operatorname{Var}[Q(o,a)\mid\mathcal D]
=\hat\phi(o)^\top\Sigma_a\hat\phi(o),
\]

so uncertainty is no longer inflated merely because the feature norm became large.
This makes `lambda` and the observation-noise variance much easier to interpret.

---

# 9. BDQN warm start

## Why it is required for the main experiment

When the BLR posterior mean is initially zero,

\[
Q(o,a)=\mu_a^\top\phi_\theta(o)=0
\]

and

\[
\frac{\partial Q}{\partial\phi}=\mu_a=0.
\]

The feature extractor initially receives no useful gradient.

## Canonical sequence

1. Train the corrected DDQN.
2. Start BDQN with `--warmstart-ddqn <ddqn best.pt>`.
3. Copy the DDQN feature extractor.
4. Freeze it by default.
5. Rebuild the BLR posterior from masked n-step replay targets.
6. Evaluate both posterior-mean and one-sample-per-episode policies.

The resulting comparison asks a clean question:

> Given the same learned representation, does posterior sampling improve action
> selection over epsilon-greedy DDQN?

`--adapt-bdqn-features` is a separate ablation. It should not be used in the main BDQN
versus DDQN table because it changes two things simultaneously: representation learning
and exploration.

For `shared_bdqn`, warm-start from the corrected `shared_ddqn` checkpoint, not from the
single-UAV checkpoint, because the observation channel count differs.

---

# 10. Posterior calibration

For selected replay transitions, the code reports

\[
\hat\sigma_\epsilon^2=
\frac1N\sum_i
\left(y_i-\mu_{a_i}^\top\hat\phi_i\right)^2.
\]

This is logged as `td_residual_variance`.

Use a small predeclared screen, for example

\[
\lambda\in\{0.1,1,10\},
\]

and noise values around the measured residual scale:

\[
\sigma_\epsilon^2\in
\{0.25\hat\sigma^2,\hat\sigma^2,4\hat\sigma^2\}.
\]

Choose the setting on validation seeds using completion, completed value, uncertainty
non-collapse and seed stability. Never tune it on the final test worlds.

Shared-BDQN remains a diagnostic baseline. Its full name in the thesis should be:

> shared-posterior independent local BDQN with a global team reward.

The principal Bayesian multi-agent method is variational Bayesian-QMIX, because its
Bayesian utility head is trained through the team TD objective rather than by assigning
the same team target to independent local regressions.

---

# 11. Detection-probability-consistent belief updates

The old map always multiplied a visible cell by `0.2` after no detection. The corrected
environments use

\[
P(z=\varnothing\mid T=x)=1-p_D
\]

for a visible target-containing cell under a no-false-positive sensor. Therefore

\[
b_{t+1}(x)\propto
\begin{cases}
(1-p_D)b_t(x),&x\text{ visible and nothing detected},\\
b_t(x),&\text{otherwise}.
\end{cases}
\]

At `p_D=1`, an observed cell is almost eliminated from the unknown-target belief. At
`p_D=0.7`, it retains 30% of its previous mass before renormalization.

This correction is necessary before claiming that Bayesian RL benefits from noisy
sensing. Otherwise the policy is told an observation model inconsistent with the
simulator.

The thesis should use only the controlled pair

\[
p_D\in\{1.0,0.7\}
\]

at first. Moving targets or intermittent communication are later stress tests.

---

# 12. Tracking is now an intentional decision

The action set remains unchanged, which keeps the code modification small. `STAY` is
reinterpreted as the primitive tracking action when the UAV is inside the tracking
radius.

For each unfinished target,

\[
k_{j,t+1}=
\begin{cases}
\min(K,k_{j,t}+1),&
\text{an assigned UAV selects STAY within }r_{\mathrm{track}},\\
\max(0,k_{j,t}-d),&\text{otherwise},
\end{cases}
\]

where the default decay is `d=1`.

This changes tracking from an automatic side effect of passing near a target into a
real search-versus-track decision:

- moving may discover more area;
- staying preserves and increases tracking progress;
- leaving a target sacrifices progress.

The multi-UAV environment still permits at most one progress increment per target and
step. It assigns a tracker only among UAVs that deliberately selected STAY.

This is the minimum modification that creates genuine search-and-track behavior without
introducing a new macro-action architecture.

---

# 13. Reward choice

## Recommended main reward

Use `--reward-mode task_potential` for the thesis and retain `legacy` only as a
reproduction ablation.

The task reward is

\[
r_t^{\mathrm{task}}=
R_t^{\mathrm{complete}}
-c_{\mathrm{time}}
-c_{\mathrm{boundary}}
-c_{\mathrm{collision}}.
\]

The shaping potential is

\[
\Phi(m)=
\eta C(m)
+\zeta\sum_j v_j d_j(m)
+\kappa\sum_j v_j\frac{k_j(m)}{K},
\]

where:

- `C(m)` is coverage ratio;
- `d_j(m)` indicates that target `j` has been detected;
- `k_j/K` is normalized tracking progress;
- target values weight mission-relevant information.

The final reward is

\[
r'_t=r_t^{\mathrm{task}}
+\gamma\Phi(m_{t+1})-\Phi(m_t).
\]

At episode termination or truncation, the next potential is defined as zero. The
shaping sum telescopes:

\[
\sum_{t=0}^{T-1}\gamma^t
\left(\gamma\Phi_{t+1}-\Phi_t\right)
=-\Phi_0+\gamma^T\Phi_T.
\]

With terminal `Phi_T=0`, the difference is the constant `-Phi_0`. Therefore shaping
improves learning without changing which policy maximizes the underlying task return,
under the augmented memory-state MDP assumption.

The detection term was added to the proposed coverage-plus-progress potential because
it supplies a mathematically safe bridge between search and tracking: detecting a target
changes the potential once, while repeatedly observing it does not generate unlimited
reward.

The default scales are deliberately moderate:

- coverage potential: 5 over complete map coverage;
- detection potential: 1 per unit target value;
- progress potential: 1 per unit target value at full progress.

Completion rewards remain 4 and 12, so completion stays dominant.

## Why not keep the capped frontier bonus as the main reward?

The old bonus rewards each newly observed cell directly and caps the reward per step.
It is not generally expressible as

\[
\gamma\Phi(s')-\Phi(s),
\]

so it can change the optimal policy. In the existing results, a frontier heuristic can
obtain more shaped reward than an oracle despite completing fewer targets. That is a
warning that the scalar reward and mission objective are misaligned.

Potential-based shaping follows Ng, Harada and Russell, *Policy Invariance Under Reward
Transformations* (ICML 1999). Search-and-track MARL papers commonly combine tracking,
incremental exploration and overlap/collision terms; for example Su and Qian (Applied
Sciences 2023) define tracking reward and exploration reward from the increase in
explored area. The overlay keeps these learning signals but places them inside a
policy-invariant potential rather than treating every shaping term as an independent
objective.

References:

- Ng, Harada & Russell, ICML 1999, “Policy Invariance Under Reward Transformations”.
- Su & Qian, Applied Sciences 13(21), 11905, 2023.
- Zhao et al., Chinese Journal of Aeronautics 38(3), 2025.

---

# 14. Installation

From the extracted overlay directory:

```bash
./apply_overlay.sh /path/to/UAV-Search-Primitives_actions
cd /path/to/UAV-Search-Primitives_actions
pip install -e .
pytest -q tests/test_thesis_corrections.py
```

The installer backs up every replaced file.

---

# 15. Canonical experiment order

## Phase A: corrected deterministic baseline

For each seed 42, 43 and 44, and each `p_D` in 1.0 and 0.7:

```bash
python scripts/train_thesis_single.py \
  --algo ddqn \
  --seed 42 \
  --detection-probability 1.0 \
  --run-dir runs/thesis_v2/single/p1/ddqn_seed42
```

Then run the same with `p_D=0.7`.

For multi-UAV:

```bash
python scripts/train_thesis_multi.py \
  --algo shared_ddqn \
  --seed 42 \
  --detection-probability 1.0 \
  --run-dir runs/thesis_v2/multi/p1/shared_ddqn_seed42

python scripts/train_thesis_multi.py \
  --algo qmix_ddqn \
  --seed 42 \
  --detection-probability 1.0 \
  --run-dir runs/thesis_v2/multi/p1/qmix_ddqn_seed42
```

Do not proceed until corrected DDQN reliably beats random and is competitive with the
local-frontier baseline on completion, not only coverage.

## Phase B: frozen-feature BDQN

```bash
python scripts/train_thesis_single.py \
  --algo bdqn \
  --warmstart-ddqn runs/thesis_v2/single/p1/ddqn_seed42/best.pt \
  --seed 42 \
  --detection-probability 1.0 \
  --run-dir runs/thesis_v2/single/p1/bdqn_seed42
```

Repeat at `p_D=0.7`, using a DDQN checkpoint trained at `p_D=0.7`. Training at one
probability and evaluating at another answers a different zero-shot robustness question.

## Phase C: Bayesian-QMIX

After deterministic QMIX is stable:

```bash
python scripts/train_thesis_multi.py \
  --algo bayes_qmix_shared \
  --seed 42 \
  --detection-probability 0.7 \
  --bayes-kl-weight 0.001 \
  --run-dir runs/thesis_v2/multi/p07/bayes_qmix_shared_seed42

python scripts/train_thesis_multi.py \
  --algo bayes_qmix_independent \
  --seed 42 \
  --detection-probability 0.7 \
  --bayes-kl-weight 0.001 \
  --run-dir runs/thesis_v2/multi/p07/bayes_qmix_independent_seed42
```

Screen with three seeds. Confirm only the thesis-critical methods with seeds 42--48 and
1000 final test worlds.

---

# 16. Thesis hypotheses enabled by the correction







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







## H1: centralized credit assignment

\[
\Delta_{\mathrm{credit}}=
J_{\mathrm{QMIX-DDQN}}-J_{\mathrm{shared-DDQN}}.
\]

After mask, epsilon-clock and update-budget corrections, this difference is much more
credibly attributable to the mixer.

## H2: single-agent posterior exploration

\[
\Delta_{\mathrm{single}}(p_D)=
J_{\mathrm{BDQN}}(p_D)-J_{\mathrm{DDQN}}(p_D).
\]

Frozen features isolate posterior exploration.

## H3: Bayesian uncertainty after credit assignment

\[
\Delta_{\mathrm{Bayes-QMIX}}(p_D)=
J_{\mathrm{Bayes-QMIX}}(p_D)-J_{\mathrm{QMIX-DDQN}}(p_D).
\]

## H4: common commitment versus exploration diversity

\[
\Delta_{\mathrm{sampling}}=
J_{\mathrm{independent}}-J_{\mathrm{shared}}.
\]

Independent samples are expected to favor coverage and early detection; shared samples
may favor coherent completion and lower overlap.

## H5: uncertainty interaction

\[
\Delta_{\mathrm{DiD}}=
[J_B(0.7)-J_D(0.7)]-[J_B(1.0)-J_D(1.0)].
\]

A positive value on completion or completed value means Bayesian exploration becomes
relatively more useful under noisy sensing.

Primary metrics are completed targets, completed value, first completion time, team
coverage, overlap, collision, detection-to-completion ratio and seed variance. Shaped
reward is secondary.




































# Environnement fixe pour les scripts thesis-v2

Ce correctif vise exclusivement le pipeline canonique de la branche `correction` :

- `train_thesis_single.py`;
- `train_thesis_multi.py`;
- `evaluate_thesis_single.py`;
- `evaluate_thesis_multi.py`;
- `evaluate_thesis_baselines.py`;
- `run_thesis_suite.py`;
- `ThesisPrimitiveSearchEnv`;
- `ThesisMultiDroneLocalMemoryEnv`.

Il ne modifie pas les anciens environnements historiques.

## Installation

Depuis la racine du dépôt :

```bash
git checkout correction
python /chemin/vers/apply_thesis_fixed_scenario.py .
pip install -e .
pytest -q tests/test_thesis_corrections.py tests/test_thesis_automation.py
```

Chaque fichier modifié reçoit une sauvegarde avec le suffixe
`.before_fixed_scenario`.

## Vérifier le plan complet

```bash
python scripts/run_thesis_suite.py fixed \
  --scenario-seed 12345 \
  --device cuda \
  --dry-run
```

## Lancer toutes les policies sur le même environnement

```bash
python scripts/run_thesis_suite.py fixed \
  --scenario-seed 12345 \
  --device cuda
```

Le preset `fixed` utilise par défaut :

- seeds d'entraînement `42,43,44,45,46`;
- un seul monde défini par `scenario_seed=12345`;
- `pD=1.0`;
- 1000 épisodes d'entraînement;
- DDQN et BDQN single-UAV;
- shared-DDQN, shared-BDQN, QMIX-DDQN et les deux Bayesian-QMIX;
- random/frontier/oracle single;
- random/local-frontier multi.

Les sorties sont isolées dans un stage du type :

```text
runs/thesis_v2/fixed_scenario12345/
logs/thesis_v2/fixed_scenario12345/
```

## Ce qui est réellement fixé

À chaque `reset()`, le simulateur remet son RNG sur `scenario_seed`. Sont donc
recréés à l'identique :

- le spawn du drone single;
- tous les spawns multi-UAV;
- les positions et valeurs des targets;
- la suite aléatoire du capteur à `pD < 1`;
- toute autre stochasticité de l'environnement utilisant `self.rng`.

Le paramètre `--seed` reste séparé. Il contrôle l'initialisation du réseau, le
replay sampling et l'exploration. Plusieurs seeds d'entraînement restent donc
des répétitions indépendantes face au même problème.

## Petit test avant 1000 épisodes

```bash
python scripts/run_thesis_suite.py custom \
  --stage fixed_debug_s12345 \
  --fixed-scenario \
  --scenario-seed 12345 \
  --scope both \
  --seeds 42 \
  --probabilities 1.0 \
  --single-algos ddqn,bdqn \
  --multi-algos shared_ddqn,qmix_ddqn \
  --train-episodes 50 \
  --validation-episodes 1 \
  --final-test-episodes 20 \
  --eval-every 10 \
  --learning-starts 100 \
  --device cpu
```

## Commande single manuelle

```bash
python scripts/train_thesis_single.py \
  --algo ddqn \
  --seed 42 \
  --episodes 1000 \
  --fixed-scenario \
  --scenario-seed 12345 \
  --validation-episodes 1 \
  --run-dir runs/thesis_v2/fixed_manual/single/ddqn/seed42 \
  --device cuda
```

## Commande multi manuelle

```bash
python scripts/train_thesis_multi.py \
  --algo qmix_ddqn \
  --seed 42 \
  --episodes 1000 \
  --fixed-scenario \
  --scenario-seed 12345 \
  --validation-episodes 1 \
  --run-dir runs/thesis_v2/fixed_manual/multi/qmix_ddqn/seed42 \
  --device cuda
```

## Évaluation

Les scripts `evaluate_thesis_single.py` et `evaluate_thesis_multi.py` relisent
`run_config.json`. Il n'est donc pas nécessaire de leur repasser
`--fixed-scenario`: ils reconstruisent automatiquement le même monde.

Pour une policy déterministe, 100 répétitions du même monde produisent la même
trajectoire. L'unité statistique principale est alors la seed d'entraînement.
Les répétitions restent utiles pour :

- les baselines random;
- les tie-breaks stochastiques de frontier/local-frontier;
- les modes posterior-sample des méthodes bayésiennes.

## Interprétation

Ce test mesure la capacité d'optimisation/mémorisation, pas la généralisation.

- inférieur à random sur le monde fixe : problème d'apprentissage ou de code;
- supérieur à random mais inférieur à frontier : le réseau apprend, mais la
  structure search-and-track de l'heuristique reste plus forte;
- très bon ici mais faible avec targets aléatoires : mémorisation du scénario;
- robuste ensuite à un spawn ou layout inédit : véritable usage des observations.

L'étape suivante devra conserver les targets fixes tout en randomisant le spawn,
puis utiliser plusieurs layouts fixes avec des layouts de test tenus à l'écart.

