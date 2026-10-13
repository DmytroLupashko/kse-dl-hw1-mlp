# CLAUDE.md

- Never execute notebooks at full scale.
- Notebooks should not contain much text.

## Directories

Each section (a top-level folder with experiments: `eda/`, `validation/`, `baseline/`, `mlp/`, `ensemble/`) has:

- Experiment cards: `<section>/experiments/`
- Experiment notebooks: `<section>/notebooks/experiments/`
- Submission notebooks: `<section>/notebooks/submissions/`
- Figures: `<section>/figures/`
- Code whose subject is this section (the MLP in `mlp/`, LightGBM in `baseline/`, raw-data parsing in `eda/`): `<section>/src/`, imported as `src` by the section's own notebooks (the notebook puts the section folder on `sys.path`) and as `<section>.src.<module>` by other sections. `ashrae/` holds only project-wide plumbing no section owns.

## Experiments

- Tracking prefix: `ashrae` (MLflow experiments `ashrae-<section>-<slug>`)
- Jupyter: `.venv/bin/jupyter`
