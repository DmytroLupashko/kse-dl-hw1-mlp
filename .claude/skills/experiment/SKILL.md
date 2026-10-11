---
name: experiment
description: Create, run or report a numbered experiment card and notebook.
argument-hint: "<request> | build [NN] | report [NN]"
---

# Experiments

Headings come only from template.md.
Files: `expNN_<slug>.md`, `expNN_<slug>.ipynb`, `expNN_<slug>.png`. "Last" = highest NN.

Mode from `$ARGUMENTS`: first word `build` or `report`, otherwise a new request.

## Request

  1. Next NN = highest existing + 1. Copy template.md.
  2. Request: the user's words, verbatim. Question: one measurable sentence (ask if you can't). Plan and Setup: proposed.
  3. Stop. Wait for approval before building.

When user types in request, create experiment_NN.md using template.md. Write request as user mentioned, then question as you understood, then suggest plan. 

## Build [NN]

  1. Build the notebook for experiment NN (default: last).
  2. Notebook holds code only: title linking the card, section headers, one-line comments on choices that aren't obvious.
  3. The config cell reads `SMOKE = os.environ.get("SMOKE") == "1"` and sets every constant that controls run size in one `if SMOKE: ... else: ...` block. Smoke mode shrinks each one (data size, iterations, folds, search grid, and so on) to the smallest value that still runs every code path, so the run takes seconds. It logs to `<experiment>-smoke` and saves no figures.
  4. Verify with `SMOKE=1 .venv/bin/jupyter nbconvert --execute --to notebook --stdout <nb> > /dev/null`. Fix until it passes. Never run at full scale and never write outputs to the notebook.


## Report [NN]

  1. Run = code cells have outputs. Otherwise ask the user to run it and stop.
  2. Results: a figure for a trend or a table for a few values (not both for the same numbers, prefer figure), plus one line saying the metric and the rows it was scored on.
  3. AI Summary: up to 2 short sentences answering the Question. Leave Conclusion to the user.
