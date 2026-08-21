import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split
from src.house_price_prediction.evaluation.metrics import MetricsCalculator
from src.house_price_prediction.evaluation.explainability import ModelExplainer
from src.house_price_prediction.utils.logger import logger
from src.house_price_prediction.utils.helpers import load_artifact
from src.house_price_prediction.utils.config import config

def main():
    logger.info("Starting model evaluation...")
    
    try:
        # Load data
        data_path = config.get(
            'data.processed_path', 
            'data/processed/house_prices_processed.csv'
        )

        df = pd.read_csv(data_path)
        
        target = config.get(
            'features.target',
            'PRICE'
        )

        numeric_features = config.get(
            'features.numeric_features',
            ['SQFT', 'BEDROOMS']
        )

        categorical_features = config.get(
            'features.categorical_features',
            [
                'LOCATION',
                'REGION',
                'TITLED',
                'LEASE',
                'FOOTINGS'
            ]
        )

        model_features = (
            numeric_features +
            categorical_features
        )

        X = df[model_features].copy()
        y = df[target]

        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=config.get(
                'preprocessing.test_size',
                0.2
            ),
            random_state=config.get(
                'preprocessing.random_state',
                42
            )
        )

        best_model_info = load_artifact(
            'artifacts/best_model_info.joblib'
        )

        model_name = best_model_info[
            'best_model'
        ]

        model = load_artifact(
            f'artifacts/trained_models/{model_name}.joblib'
        )

        preprocessor = load_artifact(
            'artifacts/preprocessor.joblib'
        )

        feature_columns = load_artifact(
            'artifacts/feature_columns.joblib'
        )

        X_test_processed = (
            preprocessor.transform(X_test)
        )

        y_pred = model.predict(
            X_test_processed
        )

        logger.info(
            f"Test observations: {len(y_test)}"
        )

        logger.info(
            f"Predictions: {len(y_pred)}"
        )

        metrics_calc = MetricsCalculator()

        metrics = (
            metrics_calc.calculate_all_metrics(
                y_test,
                y_pred
            )
        )

        metrics_df = pd.DataFrame(
            [metrics]
        )

        metrics_path = Path(
            "reports/metrics/evaluation_metrics.csv"
        )

        metrics_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        metrics_df.to_csv(
            metrics_path,
            index=False
        )

        logger.info(
            f"Metrics saved to {metrics_path}"
        )

        explainer = ModelExplainer(
            model,
            feature_columns
        )

        feature_importance = (
            explainer.get_global_importance()
        )

        feature_importance_path = Path(
            "reports/metrics/feature_importance.csv"
        )

        feature_importance.to_csv(
            feature_importance_path,
            index=False
        )

        Path(
            "reports/figures"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        plt.figure(
            figsize=(10, 8)
        )

        importance_df = (
            feature_importance.head(15)
        )

        sns.barplot(
            data=importance_df,
            x='importance',
            y='feature'
        )

        plt.title(
            f"Top 15 Feature Importance - {model_name}"
        )

        plt.tight_layout()

        plt.savefig(
            "reports/figures/feature_importance.png",
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        residuals = (
            y_test.to_numpy()
            - y_pred
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5)
        )

        axes[0].scatter(
            y_pred,
            residuals,
            alpha=0.5
        )

        axes[0].axhline(
            y=0,
            linestyle='--'
        )

        axes[0].set_xlabel(
            'Predicted Values'
        )

        axes[0].set_ylabel(
            'Residuals'
        )

        axes[0].set_title(
            'Residual Plot'
        )

        axes[1].hist(
            residuals,
            bins=30,
            edgecolor='black'
        )

        axes[1].set_xlabel(
            'Residuals'
        )

        axes[1].set_ylabel(
            'Frequency'
        )

        axes[1].set_title(
            'Residual Distribution'
        )

        plt.tight_layout()

        plt.savefig(
            "reports/figures/residual_plots.png",
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        logger.info(
            "Model evaluation completed successfully"
        )

        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)

        print(
            f"Best model: {model_name}"
        )

        for metric, value in metrics.items():

            print(
                f"{metric:20s}: {float(value):.4f}"
            )

        print("=" * 60)


    except Exception as e:

        logger.error(
            f"Error in model evaluation: {str(e)}"
        )

        raise


if __name__ == "__main__":
    main()