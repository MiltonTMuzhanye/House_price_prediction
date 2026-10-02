import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ..utils.logger import logger
from ..utils.config import config
from ..utils.helpers import save_artifact, load_artifact


class DataPreprocessor:
    """Handles preprocessing and feature engineering."""

    def __init__(self):
        self.numeric_features = config.get(
            "features.numeric_features",
            ["SQFT", "BEDROOMS"],
        )

        self.categorical_features = config.get(
            "features.categorical_features",
            ["LOCATION", "REGION", "TITLED", "LEASE", "FOOTINGS"],
        )

        self.target = config.get(
            "features.target",
            "PRICE",
        )

        self.preprocessor = None

    def map_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preserve categorical codes from the source dataset.

        The source data contains coded categorical values, including
        special codes such as 9. Without an official codebook, these
        values must not be assigned invented semantic labels.
        """

        df_mapped = df.copy()

        # Keep categorical variables as categorical/string values.
        # This prevents unknown source codes from becoming NaN.
        for column in self.categorical_features:
            if column in df_mapped.columns:
                df_mapped[column] = df_mapped[column].astype(str)

        logger.info("Categorical codes preserved from source dataset.")

        return df_mapped

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features."""

        df_engineered = df.copy()

        if "SQFT" in df_engineered.columns and "BEDROOMS" in df_engineered.columns:
            df_engineered["BEDROOMS_PER_SQFT"] = (
                df_engineered["BEDROOMS"]
                / df_engineered["SQFT"]
                * 1000
            )

        if "SQFT" in df_engineered.columns:
            df_engineered["LOG_SQFT"] = (
                df_engineered["SQFT"]
                .clip(lower=0)
                .apply(lambda x: __import__("numpy").log1p(x))
            )

        logger.info(
            f"Engineered features: {df_engineered.columns.tolist()}"
        )

        return df_engineered

    def create_preprocessing_pipeline(self):
        """Create the scikit-learn preprocessing pipeline."""

        numeric_transformer = Pipeline(
            steps=[
                ("scaler", StandardScaler())
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                (
                    "onehot",
                    OneHotEncoder(
                        drop="first",
                        sparse_output=False,
                        handle_unknown="ignore",
                    ),
                )
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_transformer,
                    self.numeric_features,
                ),
                (
                    "cat",
                    categorical_transformer,
                    self.categorical_features,
                ),
            ],
            remainder="drop",
        )

        return self.preprocessor

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit preprocessing pipeline and transform training data."""

        if self.preprocessor is None:
            self.create_preprocessing_pipeline()

        X_processed = self.preprocessor.fit_transform(X)

        save_artifact(
            self.preprocessor,
            "artifacts/preprocessor.joblib",
        )

        logger.info(
            f"Preprocessor fitted. "
            f"Input features: {len(X.columns)}, "
            f"transformed features: {X_processed.shape[1]}"
        )

        return X_processed

    def transform(self, X: pd.DataFrame):
        """Transform new data using the saved preprocessor."""

        preprocessor = load_artifact(
            "artifacts/preprocessor.joblib"
        )

        return preprocessor.transform(X)

    def get_feature_names(self):
        """Return transformed feature names."""

        if self.preprocessor is None:
            self.preprocessor = load_artifact(
                "artifacts/preprocessor.joblib"
            )

        return self.preprocessor.get_feature_names_out().tolist()
