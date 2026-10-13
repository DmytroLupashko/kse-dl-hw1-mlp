"""MLflow bookkeeping shared by every section: one `Experiment` per notebook run, tagging every run with that run's
session and notebook. Free of torch; training MLP runs: mlp/src/tracking.py."""
from time import strftime

import mlflow

from .config import MLFLOW_URI


def log_run(experiment, run_name, *, params, metrics, tags=None):
    """Logs one finished run to `experiment`, for models trained outside Lightning (no torch needed)."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name, tags=tags):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


def session_runs(name, session, **tags):
    """Finished runs of one session of experiment `name` (the latest one if `session` is None) matching `tags`,
    with the `tags.` prefix dropped from column names."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    runs = mlflow.search_runs(experiment_names=[name], filter_string="attributes.status = 'FINISHED'")
    runs = runs.rename(columns=lambda c: c.removeprefix("tags."))

    runs = runs[runs.session == (session or runs.session.max())]
    for key, value in tags.items():
        runs = runs[runs[key] == value]
    return runs


class Experiment:
    def __init__(self, name, notebook=None, session=None):
        """`notebook` tags every run; `session` reuses an earlier execution's runs instead of starting a new one."""
        self.name = name
        self.notebook = notebook
        self.reused = session is not None
        self.session = session or strftime("%Y-%m-%d %H:%M:%S")  # tag shared by this execution's runs

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(name)
        self.client = mlflow.MlflowClient()
        print(f"MLflow: {MLFLOW_URI}, experiment '{name}', session {self.session}")

    def log_run(self, run_name, *, params, metrics, tags=None):
        """Logs one finished run of a model trained outside `fit`, tagged like the others."""
        log_run(self.name, run_name, params=params, metrics=metrics, tags=self._tags(tags))

    def _tags(self, tags):
        own = {"session": self.session}
        if self.notebook:
            own["notebook"] = self.notebook
        return {**own, **(tags or {})}

    def log_metrics(self, run_id, metrics):
        for key, value in metrics.items():
            self.client.log_metric(run_id, key, value)

    def history(self, run_id, key):
        """Per-epoch values of a metric, in epoch order."""
        values = sorted(self.client.get_metric_history(run_id, key), key=lambda m: m.step)
        return [m.value for m in values]

    def runs(self):
        """This session's runs (of this notebook, if given), as a DataFrame.

        A reused session is found by its timestamp alone, so a renamed notebook still finds its earlier runs."""
        query = f"tags.session = '{self.session}'"
        if self.notebook and not self.reused:
            query += f" and tags.notebook = '{self.notebook}'"
        return mlflow.search_runs(experiment_names=[self.name], filter_string=query)
