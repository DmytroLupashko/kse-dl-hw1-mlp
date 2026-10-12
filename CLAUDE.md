# CLAUDE.md

- Never execute notebooks at full scale.
- Notebooks should not contain much text.

## Directories

Each section (a top-level folder with experiments: `eda/`, `validation/`, `baseline/`, `mlp/`, `ensemble/`) has:

- Experiment cards: `<section>/experiments/`
- Experiment notebooks: `<section>/notebooks/experiments/`
- Submission notebooks: `<section>/notebooks/submissions/`
- Figures: `<section>/figures/`

## Experiments

- Tracking prefix: `ashrae` (MLflow experiments `ashrae-<section>-<slug>`)
- Jupyter: `.venv/bin/jupyter`
