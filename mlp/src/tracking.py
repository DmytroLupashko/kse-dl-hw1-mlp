"""MLflow bookkeeping for the MLP: the project-wide `Experiment`, plus `fit`, which trains a network as one run."""
from lightning.pytorch.loggers import MLFlowLogger

from ashrae.config import MLFLOW_URI
from ashrae.tracking import Experiment as BaseExperiment

from .training import train


class Experiment(BaseExperiment):
    def fit(self, make_net, train_loader, val_loader=None, *, run_name, epochs, tags=None, params=None, callbacks=(),
            **model_kw):
        """`training.train`, logged as one run. Returns the model and its run id."""
        logger = MLFlowLogger(self.name, tracking_uri=MLFLOW_URI, run_name=run_name, tags=self._tags(tags))
        logger.log_hyperparams({
            "epochs": epochs,
            "batch_size": train_loader.batch_size,
            "n_inputs": train_loader.dataset.x.shape[1],
            **(params or {}),
        })

        model = train(make_net, train_loader, val_loader, epochs=epochs, logger=logger, desc=run_name,
                      callbacks=callbacks, **model_kw)
        return model, logger.run_id
