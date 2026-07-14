# Multi-UAV local-memory and CTDE plan

## 7. Freeze the single-UAV version

The confirmed source is formulation `D`:

- boundary action mask;
- tracking-progress map;
- tracking-progress rewards `0.30 / 0.90`;
- unchanged detection and completion rewards.

Create a release tag before changing the multi-UAV code:

```bash
git tag -a single-uav-v1 -m "Frozen single-UAV formulation D"
git push origin single-uav-v1
```

The immutable preset is `src/uav_search_belief20/experiments/frozen_single.py`.

## 8. Create MultiDroneLocalMemoryEnv

Each UAV owns a different `DroneMemory`. Belief, visited cells, known targets,
tracking progress, and completion maps are never broadcast. Teammates appear only
inside `teammate_visibility_radius`.

The team frontier reward uses an environment-only union map, so duplicate physical
coverage is not rewarded twice. The union map is not part of any local observation.
A target receives at most one tracking increment per environment step.

The compact privileged QMIX state contains UAV positions and target truth as a
31-value vector for 3 UAVs and 4 targets. It is legal during centralized training
but unavailable to `act()`.

## 9. Test shared local DDQN and BDQN

```bash
python scripts/run_multi_local_suite.py shared-screen --device cuda
```

This runs `shared_ddqn` and `shared_bdqn` on seeds 42–44. Shared parameters do not
mean shared maps. Shared BDQN uses one posterior sample for the entire team and
entire episode; call it **shared-posterior local BDQN**.

## 10. Adapt QMIX-DDQN

```bash
python scripts/run_multi_local_suite.py qmix-screen --device cuda
```

The new joint replay stores local observations, compact states, joint actions, and
current/next valid-action masks. The DDQN target performs its argmax only over valid
next actions.

## 11. Validate CTDE

```bash
python scripts/run_multi_local_suite.py smoke
```

The validation checks:

- local memory isolation;
- no target-map broadcast;
- correct boundary masks;
- one tracking increment per target and step;
- reward-part sum;
- `act()` has no global-state argument;
- changing/forbidding the mixer does not affect decentralized action selection;
- invalid actions are excluded from the target argmax;
- QMIX monotonicity;
- gradient flow through utility and mixer networks.

## Automatic experiment sequence

All three methods, 3-seed screening:

```bash
python scripts/run_multi_local_suite.py screen --device cuda
```

Seven-seed, 1000-training/1000-evaluation confirmation:

```bash
python scripts/run_multi_local_suite.py confirm --device cuda
```

Main outputs:

```text
logs/multi_local/screen/aggregate/multi_summary.md
logs/multi_local/screen/aggregate/paired_qmix_vs_shared_ddqn.csv
logs/multi_local/confirm/aggregate/multi_summary.md
```

## Thesis comparison

| Method | Local maps | Shared parameters | Central mixer in training | Mixer in execution |
|---|---:|---:|---:|---:|
| shared-DDQN local | yes | yes | no | no |
| shared-BDQN local | yes | yes | no | no |
| QMIX-DDQN local | yes | yes | yes | no |

Therefore the comparison isolates centralized credit assignment rather than map
sharing.

Prioritize completed targets/value, team coverage, coverage overlap, collisions,
detection-to-completion ratio, time to detection/completion, knowledge overlap, and
reward decomposition. Do not select a method from scalar reward alone.
