# Gantt — objectif mi/fin juillet

```mermaid
gantt
    title Plan accéléré jusqu'à fin juillet
    dateFormat  YYYY-MM-DD
    axisFormat  %d/%m

    section Repo primitive-action
    Création nouveau repo proche ancien          :done, a1, 2026-07-09, 1d
    Env primitive stay/up/down/left/right        :a2, 2026-07-09, 2026-07-10
    DQN/DDQN/BDQN agents + smoke tests           :a3, 2026-07-10, 2026-07-12

    section Single drone
    Runs DQN premières seeds                     :b1, 2026-07-12, 2026-07-14
    Runs DDQN premières seeds                    :b2, 2026-07-14, 2026-07-15
    Runs BDQN premières seeds                    :b3, 2026-07-15, 2026-07-17
    Analyse rapide et correction rewards         :b4, 2026-07-17, 2026-07-18
    Campagne single propre 5 seeds               :b5, 2026-07-18, 2026-07-21

    section Multi-drone baseline
    Env 3 drones shared memory                   :c1, 2026-07-21, 2026-07-22
    Shared BDQN independent                      :c2, 2026-07-22, 2026-07-25
    Eval baseline multi                          :c3, 2026-07-25, 2026-07-26

    section QMIX
    Replay buffer épisodique                     :d1, 2026-07-26, 2026-07-27
    QMIX-DDQN learner minimal                    :d2, 2026-07-27, 2026-07-30
    Eval QMIX-DDQN                               :d3, 2026-07-30, 2026-07-31

    section Stretch
    Prototype QMIX-BDQN                          :e1, 2026-08-01, 2026-08-04
```

## Version réaliste

- Mi-juillet : single-drone DQN/DDQN/BDQN propre.
- Fin juillet : shared BDQN 3 UAV + premier QMIX-DDQN.
- QMIX-BDQN : bonus, pas chemin critique.
