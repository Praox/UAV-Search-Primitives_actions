# Installation

This patch targets commit `2de9426fb6340be701223c980c1b027485b8cf4f`.

Copy it over the repository root:

```bash
unzip -o uncertainty_study_automation_patch.zip -d UAV-Search-Primitives_actions
cd UAV-Search-Primitives_actions
pip install -e .
```

Start with:

```bash
python scripts/run_uncertainty_study.py smoke --device cpu
```

Recommended screen:

```bash
python scripts/run_uncertainty_study.py full-screen \
  --device cuda \
  --probabilities 1.0,0.7 \
  --bayes-kl-weight 0.001
```

Read `docs/UNCERTAINTY_STUDY.md` before launching the confirmation stage.
