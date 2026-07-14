# Single-UAV finalization patch


Copy the archive contents over the repository root, then reinstall editable mode:

```bash
unzip -o single_uav_definitive_patch.zip -d UAV-Search-Primitives_actions
cd UAV-Search-Primitives_actions
pip install -e .
```

Validate the installation:

```bash
python scripts/run_single_suite.py smoke
```

Aggregate the seven existing DDQN and corrected-BDQN seeds:

```bash
python scripts/run_single_suite.py existing
```

Run the five-environment screening study tonight:

```bash
python scripts/run_single_suite.py screen --device cuda
```

This launches, sequentially and with automatic resume:

1. the smoke test;
2. random/frontier/oracle baselines;
3. DDQN for variants `v3,A,B,C,D` on seeds `42,43,44`;
4. final evaluation for every run;
5. CSV and Markdown aggregation.

Results are written to:

```text
logs/single_suite/screen/aggregate/single_summary.md
logs/single_suite/screen/aggregate/single_summary.csv
```

After selecting the strongest environment formulation, run the definitive DDQN/BDQN comparison. Example:

```bash
python scripts/run_single_suite.py confirm \
  --device cuda \
  --variants v3,C,D
```

Use `--force` to rerun completed jobs. Without `--force`, existing valid evaluation JSON files are skipped and existing checkpoints are reused.

See `docs/SINGLE_UAV_FINALIZATION.md` for the ablations, expected interpretation, and recommended evening workflow.
