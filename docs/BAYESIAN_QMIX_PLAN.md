# Steps 12--13: Bayesian-QMIX and posterior-sampling ablation

## Scientific objective

The validated comparison currently contains:

- shared-DDQN with local maps;
- shared-BDQN with local maps;
- QMIX-DDQN with local utilities and a centralized mixer.

The next controlled question is not simply whether a Bayesian final layer can be
inserted into QMIX. It is:

> Does epistemic posterior sampling improve coordinated decentralized exploration,
> and should all UAVs commit to one common hypothesis or draw independent hypotheses?

The two new methods are:

| Method | Posterior distribution | Draws per episode | Mixer during execution |
|---|---:|---:|---:|
| `bayes_qmix_shared` | one shared variational posterior | one draw copied to every UAV | no |
| `bayes_qmix_independent` | the same shared variational posterior | one independent draw per UAV | no |

The number of learned parameters is identical. Only the posterior-sampling semantics
change, which makes the comparison interpretable.

---

## Why the implementation is not a naive BLR reconstruction

In independent BDQN, a transition has a scalar Bellman target that can directly train
the selected local action head.

In QMIX, the valid target is a team value:

\[
y_{tot}=r+\gamma Q_{tot}^{target}(s',\mathbf a').
\]

There is no unique decomposition of `y_tot` into separate targets `y_i` for every
local utility. Rebuilding one Bayesian linear regression per agent against the team
target would therefore impose an arbitrary and scientifically unjustified credit
assignment.

The patch instead uses a mean-field Gaussian posterior over the final utility layer:

\[
q(W)=\prod_j \mathcal N(\mu_j,\sigma_j^2).
\]

A sampled utility head produces local utilities, the deterministic monotonic mixer
produces `Q_tot`, and the whole model is optimized end-to-end:

\[
\mathcal L =
\operatorname{MSE}(Q_{tot},y_{tot})
+\beta\frac{KL(q(W)\Vert p(W))}{N_W}.
\]

This preserves centralized credit assignment through the mixer and allows gradients
to update the feature extractor, posterior mean, posterior variance, and mixer.

The posterior is approximate rather than an exact Bayesian linear-regression
posterior. The method should be described as **variational Bayesian-QMIX with a
Bayesian last utility layer**.

---

## Training and execution

### Training

For each joint replay transition:

1. encode every local observation with one parameter-shared feature network;
2. sample utility-head weights according to the selected sampling mode;
3. compute local selected utilities `Q_i(o_i,a_i)`;
4. mix them with the centralized state into `Q_tot`;
5. compute a DDQN target using posterior-mean online utilities for selection and
   posterior-mean target utilities for evaluation;
6. optimize TD loss plus normalized KL regularization.

Posterior-mean targets are intentional: they reduce target variance while posterior
samples remain responsible for exploration and variational learning.

### Execution

At the beginning of each episode:

- shared mode draws one head and gives the same head to all UAVs;
- independent mode draws one head per UAV from the same posterior.

Every UAV then executes:

\[
a_i=\arg\max_a Q_i(o_i,a;W_i).
\]

`act()` has no global-state argument and never invokes the mixer.

---

## Fair evaluation protocol

Each Bayesian checkpoint is evaluated three ways on exactly the same world seeds:

1. posterior sampling shared by all UAVs;
2. posterior sampling independently per UAV;
3. deterministic posterior-mean execution.

The sampling mode used during training remains the primary row in the final JSON.
Cross-execution metrics are stored with `sampled_shared_` and
`sampled_independent_` prefixes. Mean-policy metrics use `mean_policy_`. The file also
contains `execution_independent_minus_shared_*` deltas, which isolate the execution
sampling effect for one fixed trained posterior.

Checkpoint selection during training uses posterior-mean evaluation. This avoids
selecting a checkpoint only because it received a lucky evaluation draw.

A separate posterior-evaluation RNG seed is used so final Thompson evaluations do not
depend on how many random numbers were consumed during training.

---

## Efficient experiment process

### Phase 12A -- Contract validation

```bash
python scripts/run_multi_local_suite.py bayes-smoke --device cpu
```

The cumulative test checks:

- independent local memories and no target broadcast;
- action masks and reward decomposition;
- at most one tracking increment per target and step;
- compact state shapes;
- shared posterior samples are exactly identical across UAVs;
- independent samples differ;
- mixer absence during decentralized `act()`;
- valid-action target argmax;
- finite Bayesian-QMIX update;
- gradients through feature network, posterior head, and mixer;
- positive bounded posterior standard deviations;
- checkpoint save/load;
- instantaneous sensor-overlap metric.

### Phase 12B -- Calibrate only the Bayesian regularization

```bash
python scripts/run_multi_local_suite.py bayes-kl-screen --device cuda
```

Use shared sampling only to select one global KL value. Keep prior standard deviation
and initial posterior standard deviation fixed. This prevents a large combinatorial
hyperparameter search.

Reject a KL value when:

- posterior standard deviation collapses immediately to the minimum;
- posterior standard deviation grows to the maximum;
- reward variance explodes;
- completion degrades despite apparently high uncertainty.

### Phase 13A -- Isolate sampling semantics

```bash
python scripts/run_multi_local_suite.py bayes-screen --device cuda \
  --bayes-kl-weight <selected_value>
```

The shared and independent methods must use:

- identical architecture;
- identical KL weight and posterior initialization;
- identical training/evaluation worlds;
- identical checkpoint rule;
- identical environment and mixer state.

Primary interpretation metrics:

1. completed targets and completed value;
2. team coverage;
3. cumulative coverage overlap;
4. simultaneous sensor overlap;
5. collisions;
6. first detection and first completion;
7. seed variance;
8. posterior standard deviation and sample distance.

### Phase 13B -- Confirm and extend seeds

```bash
python scripts/run_multi_local_suite.py bayes-confirm --device cuda \
  --bayes-kl-weight <selected_value>
```

Use Student-t intervals from the new aggregator. When completion or completed-value
intervals still straddle zero, append seeds 49--55 rather than retuning the model.

---

## How the other proposed experiments align with steps 12--13

### 1. Increased number of seeds

Do this after the 7-seed Bayesian confirmation, not before the KL screen. Additional
seeds reduce estimator uncertainty; they should not be used to search hyperparameters.

### 2. Noisy detection

Run only after the deterministic posterior-sampling comparison is frozen:

```bash
python scripts/run_multi_local_suite.py bayes-noise-screen --device cuda
```

This tests the hypothesis that epistemic exploration becomes more useful when local
observations are less reliable. Detection probability changes environment stochasticity,
so all three QMIX methods are retrained at every probability.

### 3. Simultaneous overlap

This metric is added immediately because it is cheap and observational only. It is the
fraction of the current union sensor footprint that is observed by more than one UAV
at the same step. It separates momentary coordination from cumulative revisit history.

### 4. Privileged versus memory-union mixer state

The two states have the same length:

- `privileged_truth`: hidden target positions, values, detection/completion and progress;
- `memory_union`: the same target slots, but filled only from records existing in at
  least one local memory; unknown targets are zeros.

Because dimensions are equal, a performance difference is attributable mainly to
information content rather than mixer capacity.

### 5. Recurrent QMIX

Add recurrence after steps 12--13 are frozen. Otherwise recurrence and posterior
sampling change simultaneously and the cause of any gain becomes ambiguous.

Recommended later factorial comparison:

| Utility memory | Posterior sampling |
|---|---|
| feed-forward | deterministic |
| feed-forward | shared Bayesian |
| feed-forward | independent Bayesian |
| GRU | deterministic |
| GRU | shared Bayesian |
| GRU | independent Bayesian |

First implement sequence replay and burn-in for deterministic recurrent QMIX. Only
then reuse the same variational last-layer module.

---

## Expected scientific signatures

### Shared sampling is preferable when

- UAVs coordinate on complementary actions despite one common hypothesis;
- collisions and simultaneous overlap decrease;
- seed variance remains controlled;
- the team avoids the incoherent independent exploration of the same target.

### Independent sampling is preferable when

- team coverage increases;
- first detection becomes earlier;
- knowledge overlap decreases;
- one bad posterior draw no longer traps the whole team;
- completion does not fall due to excessive behavioral diversity.

### Bayesian-QMIX is useful only if

- sampled-policy gains are also visible, at least partly, under posterior-mean
  evaluation or across many posterior draws;
- uncertainty remains non-degenerate;
- gains persist across seeds and noisy scenarios;
- improvements are not explained only by a different collision or STAY pathology.

---

## Output interpretation

The most useful files are:

```text
aggregate/multi_summary.csv
aggregate/paired_bayes_shared_vs_independent.csv
aggregate/paired_bayes_shared_vs_qmix.csv
aggregate/paired_bayes_independent_vs_qmix.csv
aggregate/paired_all_long.csv
```

`multi_summary.csv` contains both Student-t and z intervals. For seven seeds, report
Student-t intervals in the thesis.

Training `metrics.csv` additionally logs:

- total, TD, and KL losses;
- posterior mean/max standard deviation;
- KL per posterior parameter;
- normalized distance between episode samples;
- simultaneous overlap.
