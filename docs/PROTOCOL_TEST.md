Protocole de validation du learning

Ce protocole cherche à répondre à une question simple :

La politique apprise progresse-t-elle réellement sur des scénarios non vus,et dépasse-t-elle une politique aléatoire utilisant exactement les mêmesaction masks ?

Les métriques principales sont completed, completed_value etfirst_completion_step. La reward façonnée ne doit pas être utilisée seulecomme preuve d'apprentissage.

0. Installation des fichiers

Remplacer :

src/uav_search_belief20/envs/thesis_envs.py
scripts/train_thesis_single.py
scripts/train_thesis_multi.py

Puis :

pip install -e .
python -m py_compile \
  src/uav_search_belief20/envs/thesis_envs.py \
  scripts/train_thesis_single.py \
  scripts/train_thesis_multi.py

Ne pas réutiliser un checkpoint entraîné avec les cartes globales commewarm-start d'un modèle égocentrique.

1. Tests fonctionnels obligatoires

1.1 Action mask single

À la position (0, 0), le mask attendu dans l'ordreSTAY, UP, DOWN, LEFT, RIGHT est :

[True, False, True, False, True]

Exécuter :

python - <<'PY'
import numpy as np
from uav_search_belief20.envs.thesis_envs import (
    ThesisEnvConfig,
    ThesisPrimitiveSearchEnv,
)

env = ThesisPrimitiveSearchEnv(
    ThesisEnvConfig(
        seed=0,
        use_boundary_action_mask=True,
        include_track_progress_map=True,
        observation_frame="egocentric",
    )
)
env.drone_pos = np.array([0, 0], dtype=np.int64)
assert env.action_mask().tolist() == [True, False, True, False, True]
print("single action mask: OK")
PY

1.2 Action mask multi

python - <<'PY'
import numpy as np
from uav_search_belief20.envs.thesis_envs import (
    ThesisMultiDroneLocalMemoryEnv,
    ThesisMultiEnvConfig,
)

env = ThesisMultiDroneLocalMemoryEnv(
    ThesisMultiEnvConfig(
        n_agents=3,
        seed=0,
        use_boundary_action_mask=True,
        observation_frame="egocentric",
    )
)
env.drone_pos = np.array([[0, 0], [19, 19], [10, 10]], dtype=np.int64)
masks = env.action_mask()
assert masks[0].tolist() == [True, False, True, False, True]
assert masks[1].tolist() == [True, True, False, True, False]
assert masks[2].tolist() == [True, True, True, True, True]
print("multi action masks: OK")
PY

1.3 UAV toujours au centre

python - <<'PY'
from uav_search_belief20.envs.thesis_envs import (
    ThesisEnvConfig,
    ThesisPrimitiveSearchEnv,
)

env = ThesisPrimitiveSearchEnv(
    ThesisEnvConfig(
        seed=1,
        include_track_progress_map=True,
        use_boundary_action_mask=True,
        observation_frame="egocentric",
    )
)
obs, _ = env.reset()
anchor = env.cfg.grid_size // 2
assert obs.shape == (7, 20, 20)
assert obs[0, anchor, anchor] == 1.0

for _ in range(50):
    mask = env.action_mask()
    action = int(next(index for index, allowed in enumerate(mask) if allowed))
    obs, _, terminated, truncated, _ = env.step(action)
    assert obs[0, anchor, anchor] == 1.0
    if terminated or truncated:
        break

print("egocentric single observation: OK")
PY

1.4 Translation relative

Créer deux états ayant le même déplacement relatif UAV-cible, par exemple :

UAV A = (2, 2), cible A = (2, 5)
UAV B = (12, 10), cible B = (12, 13)

Dans les deux observations, la cible doit apparaître exactement au même pixellocal. Ce test doit porter sur les canaux known_target_value ettrack_progress.

1.5 STAY réellement requis

Construire un état où une cible connue est à distance inférieure ou égale àtrack_radius.

Vérifier :

STAY     -> progression +1
mouvement -> aucune progression
interruption -> décroissance selon track_progress_decay

1.6 Potentiel

Sur une carte 20 x 20, une nouvelle cellule couverte modifie le terme decouverture de :

10 / 400 = 0.025

À état identique pour la détection et le tracking :

python - <<'PY'
from uav_search_belief20.envs.thesis_envs import (
    ThesisEnvConfig,
    ThesisPrimitiveSearchEnv,
)

env = ThesisPrimitiveSearchEnv(
    ThesisEnvConfig(
        seed=0,
        coverage_potential_scale=10.0,
        detection_potential_scale=1.0,
        progress_potential_scale=2.0,
    )
)
env.reset()
before = env._potential()
cell = next(
    (r, c)
    for r in range(env.cfg.grid_size)
    for c in range(env.cfg.grid_size)
    if env.memory.visited[r, c] < 0.5
)
env.memory.visited[cell] = 1.0
after = env._potential()
assert abs((after - before) - 0.025) < 1e-6
print("coverage potential: OK")
PY

2. Baseline random masquée

La baseline random doit tirer uniformément parmi les actions autorisées :

allowed = np.flatnonzero(env.action_mask())
action = rng.choice(allowed)

Pour le multi, faire ce tirage indépendamment pour chaque UAV à partir de sonpropre mask.

Évaluer au minimum 1000 épisodes sur les seeds 200000 ... 200999. Enregistrer :

reward
detected
completed
completed_value
coverage
first_detection_step
first_completion_step
stay_ratio
tracking_progress_ratio
boundary_hit_ratio
collision_agent_ratio (multi)

Le boundary_hit_ratio doit être exactement nul avec les nouveaux trainers.

3. Test d'overfit contrôlé

Avant de tester la généralisation, vérifier que le réseau est capabled'apprendre un seul monde fixe. Si ce test échoue, il reste un problèmed'environnement, de reward, de réseau ou de Bellman target.

Single DDQN

python scripts/train_thesis_single.py \
  --algo ddqn \
  --seed 42 \
  --episodes 500 \
  --fixed-scenario \
  --scenario-seed 12345 \
  --observation-frame egocentric \
  --reward-mode task_potential \
  --coverage-potential-scale 10 \
  --detection-potential-scale 1 \
  --progress-potential-scale 2 \
  --eval-every 25 \
  --validation-episodes 50 \
  --skip-final-eval \
  --run-dir runs/diagnostic/single_fixed_seed42

Critère attendu :

validation_completed final > validation_completed initial
validation_completed final > random fixe
first_completion_step diminue
boundary_hit_ratio = 0
loss et Q-values restent finis

Multi shared-DDQN

Commencer par shared_ddqn, pas QMIX ni Bayesian QMIX :

python scripts/train_thesis_multi.py \
  --algo shared_ddqn \
  --seed 42 \
  --episodes 800 \
  --fixed-scenario \
  --scenario-seed 12345 \
  --observation-frame egocentric \
  --reward-mode task_potential \
  --coverage-potential-scale 10 \
  --detection-potential-scale 1 \
  --progress-potential-scale 2 \
  --global-state-mode memory_union \
  --eval-every 25 \
  --validation-episodes 50 \
  --skip-final-eval \
  --run-dir runs/diagnostic/multi_shared_fixed_seed42

Si le single ou le shared-DDQN ne peut pas sur-apprendre un monde fixe, ne paslancer BDQN/QMIX.

4. Ablation qui isole les deux corrections

Utiliser exactement les mêmes seeds et hyperparamètres.

Expérience

Repère

Potentiel

A

global

5C + D + P

B

global

10C + D + 2P

C

egocentric

5C + D + P

D

egocentric

10C + D + 2P

Cela permet de distinguer :

B - A : effet de la reward
C - A : effet du repère égocentrique
D - C : effet de la reward avec le nouveau repère
D - A : effet total

Ne comparer d'abord que DDQN single, puis shared-DDQN multi.

5. Test de généralisation single

Utiliser au minimum cinq seeds d'apprentissage :

42, 43, 44, 45, 46

Chaque seed doit utiliser les mêmes seeds de validation et de test.

Commande type :

for SEED in 42 43 44 45 46; do
  python scripts/train_thesis_single.py \
    --algo ddqn \
    --seed "$SEED" \
    --episodes 2000 \
    --observation-frame egocentric \
    --reward-mode task_potential \
    --coverage-potential-scale 10 \
    --detection-potential-scale 1 \
    --progress-potential-scale 2 \
    --eval-every 50 \
    --validation-episodes 100 \
    --final-test-episodes 1000 \
    --run-dir "runs/single_ddqn_ego_phi10_1_2/seed${SEED}"
done

6. Test de généralisation multi

Ordre recommandé :

1. shared_ddqn
2. qmix_ddqn
3. shared_bdqn avec warm-start du shared_ddqn égocentrique
4. Bayesian QMIX

Commande shared-DDQN :

for SEED in 42 43 44 45 46; do
  python scripts/train_thesis_multi.py \
    --algo shared_ddqn \
    --seed "$SEED" \
    --episodes 2500 \
    --n-agents 3 \
    --observation-frame egocentric \
    --reward-mode task_potential \
    --coverage-potential-scale 10 \
    --detection-potential-scale 1 \
    --progress-potential-scale 2 \
    --global-state-mode memory_union \
    --eval-every 50 \
    --validation-episodes 100 \
    --final-test-episodes 1000 \
    --run-dir "runs/multi_shared_ddqn_ego_phi10_1_2/seed${SEED}"
done

Commande QMIX-DDQN :

for SEED in 42 43 44 45 46; do
  python scripts/train_thesis_multi.py \
    --algo qmix_ddqn \
    --seed "$SEED" \
    --episodes 2500 \
    --n-agents 3 \
    --observation-frame egocentric \
    --reward-mode task_potential \
    --coverage-potential-scale 10 \
    --detection-potential-scale 1 \
    --progress-potential-scale 2 \
    --global-state-mode memory_union \
    --eval-every 50 \
    --validation-episodes 100 \
    --final-test-episodes 1000 \
    --run-dir "runs/multi_qmix_ddqn_ego_phi10_1_2/seed${SEED}"
done

7. Preuve que le learning croît

Ne pas conclure à partir d'une seule courbe ou d'un seul seed.

Pour chaque seed, calculer :

early = moyenne des trois premières évaluations
late  = moyenne des trois dernières évaluations
delta = late - early

Faire cela pour :

validation_completed
validation_completed_value
validation_coverage
validation_first_completion

La preuve minimale est :

moyenne(delta_completed) > 0
moyenne(delta_completed_value) > 0
late_completed > moyenne random
late_completed_value > moyenne random

La preuve forte est :

borne basse de l'IC 95 % de
(trained_final - random) > 0

La comparaison doit être faite sur les mêmes seeds d'évaluation.

8. Signaux d'échec à surveiller

Pas d'apprentissage

validation_completed reste plat au niveau random
fixed_scenario ne progresse pas
loss reste exactement nulle après learning_starts

Q-values instables

q_mean ou target_mean devient NaN/Inf
amplitude des Q-values croît sans stabilisation
loss augmente continuellement

Politique dégénérée

stay_ratio proche de 1 sans tracking_progress
stay_ratio proche de 0 malgré des cibles détectées
coverage élevée mais completed nul
tracking_progress fréquent mais completed nul

Erreur de mask

boundary_hit_ratio > 0
RuntimeError: Masked action ... selected

Une erreur fail-fast de mask est utile : elle indique qu'un chemin de sélectionou un replay n'utilise pas correctement les actions valides.

9. Critère final de validation

La correction est considérée comme réussie seulement si :

les tests fonctionnels passent ;

le DDQN single sur-apprend le scénario fixe ;

le shared-DDQN multi sur-apprend le scénario fixe ;

sur cinq seeds aléatoires, la performance finale dépasse la random masquée ;

l'amélioration porte sur completed et completed_value, pas uniquementsur la reward ou la couverture ;

les résultats restent positifs sur les 1000 seeds de test jamais utiliséspendant l'entraînement ou la sélection de checkpoint.





pip install -e .
pip install matplotlib

Vérifie la syntaxe :

python -m py_compile \
  src/uav_search_belief20/envs/thesis_envs.py \
  scripts/train_thesis_single.py \
  scripts/train_thesis_multi.py \
  scripts/plot_thesis_learning.py
2. Faire un premier entraînement single

Commence par DDQN, sur un seul seed :

python scripts/train_thesis_single.py \
  --algo ddqn \
  --seed 42 \
  --episodes 500 \
  --observation-frame egocentric \
  --reward-mode task_potential \
  --coverage-potential-scale 10 \
  --detection-potential-scale 1 \
  --progress-potential-scale 2 \
  --eval-every 25 \
  --validation-episodes 50 \
  --skip-final-eval \
  --run-dir runs/single_ddqn_ego/test/normal/seed42


python scripts/train_thesis_single.py \
  --algo ddqn \
  --seed 42 \
  --episodes 900 \
  --observation-frame global \
  --reward-mode task_potential \
  --coverage-potential-scale 10 \
  --detection-potential-scale 1 \
  --progress-potential-scale 2 \
  --eval-every 25 \
  --validation-episodes 50 \
  --skip-final-eval \
  --run-dir runs/single_ddqn_ego/test/global_task/seed42


python scripts/train_thesis_single.py \
  --algo ddqn \
  --seed 42 \
  --episodes 900 \
  --observation-frame global \
  --reward-mode legacy \
  --coverage-potential-scale 10 \
  --detection-potential-scale 1 \
  --progress-potential-scale 2 \
  --eval-every 25 \
  --validation-episodes 50 \
  --skip-final-eval \
  --run-dir runs/single_ddqn_ego/test/global_legacy/seed42


Tu obtiendras notamment :

runs/single_ddqn_ego/seed42/
├── metrics.csv
├── best.pt
├── latest.pt
├── run_config.json
└── training_status.json

Le metrics.csv contient notamment :

episode
validation_reward
validation_completed
validation_completed_value
validation_coverage
validation_first_completion
validation_tracking_progress
loss
q_mean
target_mean
epsilon

Ces colonnes sont déjà enregistrées par le trainer single.

3. Générer les courbes d’un seul run


python scripts/plot_thesis_learning.py \
  --group "DDQN centered_legacy=runs/single_ddqn_ego/test/ego_legacy/seed42" \
  --group "DDQN global_potential=runs/single_ddqn_ego/test/global_task/seed42" \
  --group "DDQN centered_potential=runs/single_ddqn_ego/test/ego_task/seed42" \
  --group "DDQN global_legacy=runs/single_ddqn_ego/test/global_legacy/seed42" \
  --output-dir figures/single_ddqn_seed42/test/all/ \
  --smooth 3 \
  --show-seeds \
  --title-prefix "Single UAV DDQN"


  4. Faire tourner plusieurs seeds

Pour conclure qu’il y a réellement apprentissage, un seul seed ne suffit pas.

for SEED in 42 43 44 45 ; do
  python scripts/train_thesis_single.py \
    --algo bdqn \
    --seed "$SEED" \
    --episodes 1500 \
    --observation-frame egocentric \
    --reward-mode legacy \
    --detection-probability 0.7 \
    --eval-every 50 \
    --validation-episodes 100 \
    --skip-final-eval \
    --run-dir "runs/refund/test/ego_legacy/p0_7/bdqn/seed${SEED}"
done

Puis génère une courbe moyenne avec les cinq seeds :

python scripts/plot_thesis_learning.py \
  --group "DDQN p0.7\=runs/refund/test/ego_legacy/p0_7/ddqn/seed*" \
  --group "BDQN p0.7\=runs/refund/test/ego_legacy/p0_7/bdqn/seed*" \
  --output-dir figures/single/test/ego_legacy/p0_7/ddqn_vs_bdqn \
  --smooth 3 \
  --show-seeds \
  --title-prefix "Single UAV"

python scripts/plot_thesis_learning.py \
  --group "BDQN ego_legacy=runs/refund/single_bdqn_ego/2nd_run/seed*" \
  --group "DDQN ego_legacy=runs/refund/single_ddqn_ego/test/ego_legacy/seed*" \
  --output-dir figures/single_BDQN_DDQN/ego_legacy/ \
  --smooth 1 \
  --show-seeds \
  --title-prefix "Single UAV"

  puis

  for SEED in 42 43 44 45 46; do
  python scripts/train_thesis_multi.py \
    --algo shared_ddqn \
    --seed "$SEED" \
    --episodes 2000 \
    --n-agents 3 \
    --observation-frame egocentric \
    --reward-mode task_potential \
    --coverage-potential-scale 10 \
    --detection-potential-scale 1 \
    --progress-potential-scale 2 \
    --global-state-mode memory_union \
    --eval-every 50 \
    --validation-episodes 100 \
    --skip-final-eval \
    --run-dir "runs/multi_shared_ddqn_ego/seed${SEED}"
done

Génère ensuite les courbes :

python scripts/plot_thesis_learning.py \
  --group "Shared-DDQN=runs/multi_shared_ddqn_ego/seed*" \
  --output-dir figures/multi_shared_ddqn_ego \
  --smooth 3 \
  --show-seeds \
  --title-prefix "Multi-UAV Shared-DDQN"

Puis QMIX :

for SEED in 42 43 44 45 ; do
  python scripts/train_thesis_multi.py \
    --algo shared_ddqn \
    --seed "$SEED" \
    --episodes 1500 \
    --n-agents 3 \
    --observation-frame egocentric \
    --reward-mode legacy \
    --detection-probability 0.5 \
    --global-state-mode memory_union \
    --eval-every 50 \
    --validation-episodes 100 \
    --skip-final-eval \
    --run-dir "runs/refund/multi_ego_leg/p0_5/shared_ddqn/seed${SEED}"
done

for SEED in 44 45 46 47 ; do
  python scripts/train_thesis_multi.py \
    --algo bayes_qmix_independent \
    --seed "$SEED"\
    --episodes 1500 \
    --n-agents 3 \
    --observation-frame egocentric \
    --reward-mode task_potential \
    --coverage-potential-scale 10 \
    --detection-potential-scale 1 \
    --progress-potential-scale 2 \
    --global-state-mode memory_union \
    --eval-every 50 \
    --validation-episodes 100 \
    --skip-final-eval \
    --run-dir "runs/refund/multi_qmix_ddqn_ego/seed${SEED}"
done
Comparaison visuelle :

python scripts/plot_thesis_learning.py \
  --group "DDQN Qmix p0_7=runs/refund/multi_ego_leg/p0_7/qmix_ddqn/seed*" \
  --group "Bayesian QMIX Independent p0_7=runs/refund/multi_ego_leg/p0_7/qmix_bdqn_indep/seed*" \
  --output-dir figures/multi_comparison/ego_legacy/p0_7/ddqn_qmix_vs_bayes_qmix_indep/ \
  --smooth 3 \
  --show-seeds \
  --title-prefix "Multi-UAV"


  Exemple single-UAV
python scripts/evaluate_thesis_checkpoint.py \
  --scope single \
  --run-dir runs/single_ddqn_ego_legacy_p1/seed42 \
  --episodes 1000 \
  --seed-base 200000 \
  --device auto


Exemple multi-UAV
python scripts/evaluate_thesis_checkpoint.py \
  --scope multi \
  --run-dir runs/multi_bayesian_qmix_ego_legacy_p07/seed42 \
  --episodes 1000 \
  --seed-base 200000 \
  --device auto

  Pour plusieurs seeds :

for RUN in runs/single_ddqn_ego_legacy_p1/seed*; do
  python scripts/evaluate_thesis_checkpoint.py \
    --scope single \
    --run-dir "$RUN" \
    --episodes 1000 \
    --seed-base 200000 \
    --device auto
done

Puis lance :

python scripts/aggregate_final_tests.py \
  --glob "runs/single_ddqn_ego_legacy_p1/seed*/final_test.json" \
  --mode mean \
  --output-dir results/single_ddqn_ego_legacy_p1

Pour BDQN sampled :

python scripts/aggregate_final_tests.py \
  --glob "runs/single_bdqn_ego_legacy_p1/seed*/final_test.json" \
  --mode sample \
  --output-dir results/single_bdqn_ego_legacy_p1_sampled

Pour le multi :

python scripts/aggregate_final_tests.py \
  --glob "runs/multi_bayesian_qmix_ego_legacy_p07/seed*/final_test.json" \
  --mode mean \
  --output-dir results/multi_bayesian_qmix_ego_legacy_p07