# Local-memory multi-UAV CTDE patch

Install over the repository root:

```bash
unzip -o multi_local_ctde_patch.zip -d UAV-Search-Primitives_actions
cd UAV-Search-Primitives_actions
pip install -e .
```

The patch adds new files and leaves the historical shared-memory environment and
historical QMIX implementation unchanged.

```bash
python scripts/run_multi_local_suite.py smoke
python scripts/run_multi_local_suite.py shared-screen --device cuda
python scripts/run_multi_local_suite.py qmix-screen --device cuda
python scripts/run_multi_local_suite.py screen --device cuda
python scripts/run_multi_local_suite.py confirm --device cuda
```

Read `docs/MULTI_LOCAL_CTDE_PLAN.md` before launching the definitive runs.
