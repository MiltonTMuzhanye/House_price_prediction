import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

from ..utils.logger import logger
from ..utils.config import config
from ..utils.helpers import save_artifact

from ..models.random_forest import RandomForestModel
from ..models.xgboost_model import XGBoostModel
from ..models.lightgbm_model import LightGBMModel
from ..models.catboost_model import CatBoostModel
from ..models.baseline import LinearRegressionModel


class ModelTrainer:
    """Handles model training and evaluation."""

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.results = {}
        self.config = config

    def initialize_models(self):
        """Initialize all models with their parameters."""

        models_config = self.config.get("model", {})

        self.models = {
            "linear_regression": LinearRegressionModel(),
            "random_forest": RandomForestModel(
                **models_config.get("random_forest_params", {})
            ),
            "xgboost": XGBoostModel(
                **models_config.get("xgboost_params", {})
            ),
            "lightgbm": LightGBMModel(
                **models_config.get("lightgbm_params", {})
            ),
            "catboost": CatBoostModel(
                **models_config.get("catboost_params", {})
            ),
        }

        models_to_train = models_config.get(
            "models_to_train",
            list(self.models.keys()),
        )

        self.models = {
            name: model
            for name, model in self.models.items()
            if name in models_to_train
        }

        logger.info(
            f"Models initialized: {list(self.models.keys())}"
        )

    def train_and_evaluate(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        """Train and evaluate all configured models."""

        cv_folds = config.get(
            "training.cv_folds",
            5,
        )

        scoring = config.get(
            "training.scoring",
            "r2",
        )

        experiment_name = config.get(
            "training.experiment_name",
            "house_price_prediction",
        )

        mlflow.set_experiment(experiment_name)

        for name, model in self.models.items():

            logger.info(f"Training {name}...")

            with mlflow.start_run(run_name=name):

                model.train(
                    X_train,
                    y_train,
                )

                predictions = model.predict(
                    X_test,
                )

                metrics = self.evaluate(
                    y_test,
                    predictions,
                )

                cv_scores = cross_val_score(
                    model.model,
                    X_train,
                    y_train,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=-1,
                )

                metrics["cv_mean"] = float(
                    np.mean(cv_scores)
                )

                metrics["cv_std"] = float(
                    np.std(cv_scores)
                )

                self.results[name] = metrics

                for metric_name, value in metrics.items():
                    mlflow.log_metric(
                        metric_name,
                        float(value),
                    )

                if hasattr(model.model, "get_params"):
                    mlflow.log_params(
                        model.model.get_params()
                    )

                model.save_model()

                logger.info(
                    f"{name} results: {metrics}"
                )

        self.select_best_model()

        return self.results

    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics."""

        return {
            "mae": mean_absolute_error(
                y_true,
                y_pred,
            ),
            "rmse": np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            ),
            "r2_score": r2_score(
                y_true,
                y_pred,
            ),
            "mape": np.mean(
                np.abs(
                    (y_true - y_pred) / y_true
                )
            ) * 100,
        }

    def select_best_model(self):
        """Select the best model based on R2 score."""

        if not self.results:
            raise ValueError(
                "No models trained yet"
            )

        best_model_name = max(
            self.results,
            key=lambda name: self.results[name]["r2_score"],
        )

        self.best_model = self.models[
            best_model_name
        ]

        best_metrics = self.results[
            best_model_name
        ]

        logger.info(
            f"Best model: {best_model_name} "
            f"with R2: "
            f"{best_metrics['r2_score']:.4f}"
        )

        save_artifact(
            {
                "best_model": best_model_name,
                "metrics": best_metrics,
            },
            "artifacts/best_model_info.joblib",
        )

        return self.best_model
