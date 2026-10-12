"""MLflow bookkeeping: one `Experiment` per notebook run, tagging every run with that run's session and notebook."""
from time import strftime

import mlflow

from .config import MLFLOW_URI


def log_run(experiment, run_name, *, params, metrics, tags=None):
    """Logs one finished run to `experiment`, for models trained outside `Experiment.fit` (no torch needed)."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=tags):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


class Experiment:
    def __init__(self, name, notebook=None, session=None):
        """`notebook` tags every run; `session` reuses an earlier execution's runs instead of starting a new one."""
        self.name, self.notebook, self.reused = name, notebook, session is not None
        self.session = session or strftime("%Y-%m-%d %H:%M:%S")  # tag shared by this execution's runs
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(name)
        self.client = mlflow.MlflowClient()
        print(f"MLflow: {MLFLOW_URI}, experiment '{name}', session {self.session}")

    def fit(self, make_net, train_loader, val_loader=None, *, run_name, epochs, tags=None, params=None, callbacks=(),
            **model_kw):
        """Seeds, builds the net, trains it for `epochs` and logs the run; `callbacks` are extra Lightning callbacks.
        Returns the model and its run id."""
        # Imported here so that `log_run` works without torch (LightGBM crashes next to it unless single-threaded)
        import lightning.pytorch as L
        from lightning.pytorch.loggers import MLFlowLogger

        from .modeling import EnergyModel, Progress

        L.seed_everything(0, verbose=False)
        model = EnergyModel(make_net(), **model_kw)
        logger = MLFlowLogger(self.name, tracking_uri=MLFLOW_URI, run_name=run_name, tags=self._tags(tags))
        logger.log_hyperparams({"epochs": epochs, "batch_size": train_loader.batch_size,
                                "n_inputs": train_loader.dataset.x.shape[1], **(params or {})})
        trainer = L.Trainer(max_epochs=epochs, logger=logger, enable_checkpointing=False,
                            enable_progress_bar=False, enable_model_summary=False, callbacks=[Progress(run_name), *callbacks])
        trainer.fit(model, train_loader, val_loader)
        return model, logger.run_id

    def log_run(self, run_name, *, params, metrics, tags=None):
        """Logs one finished run of a model trained outside `fit`, tagged like the others."""
        log_run(self.name, run_name, params=params, metrics=metrics, tags=self._tags(tags))

    def _tags(self, tags):
        return {"session": self.session, **({"notebook": self.notebook} if self.notebook else {}), **(tags or {})}

    def log_metrics(self, run_id, metrics):
        for key, value in metrics.items():
            self.client.log_metric(run_id, key, value)

    def history(self, run_id, key):
        """Per-epoch values of a metric, in epoch order."""
        return [m.value for m in sorted(self.client.get_metric_history(run_id, key), key=lambda m: m.step)]

    def runs(self):
        """This session's runs (of this notebook, if given), as a DataFrame.

        A reused session is found by its timestamp alone, so a renamed notebook still finds its earlier runs."""
        notebook = f" and tags.notebook = '{self.notebook}'" if self.notebook and not self.reused else ""
        return mlflow.search_runs(experiment_names=[self.name],
                                  filter_string=f"tags.session = '{self.session}'{notebook}")
