import numpy as np
import pandas as pd
from typing import Dict, Any

from src.house_price_prediction.utils.helpers import load_artifact
from src.house_price_prediction.utils.logger import logger


class InferencePreprocessor:
    """Preprocess input data using the fitted training preprocessor."""

    def __init__(self):
        self.preprocessor = None
        self.feature_columns = None
        self.load_preprocessor()

    def load_preprocessor(self):
        """Load the fitted preprocessing artifacts."""

        try:
            self.preprocessor = load_artifact(
                "artifacts/preprocessor.joblib"
            )

            self.feature_columns = load_artifact(
                "artifacts/feature_columns.joblib"
            )

            logger.info(
                "Loaded preprocessor and feature columns"
            )

        except Exception as e:
            logger.error(
                f"Error loading preprocessor: {str(e)}"
            )
            raise

    def _prepare_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare raw inference features to match training input."""

        df = df.copy()

        required_features = [
            "SQFT",
            "BEDROOMS",
            "LOCATION",
            "REGION",
            "TITLED",
            "LEASE",
            "FOOTINGS",
        ]

        missing_features = [
            feature
            for feature in required_features
            if feature not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}"
            )

        # Preserve categorical codes in the same numeric form
        # used when the fitted encoder was trained.

        # Match the feature engineering used during training.
        df["BEDROOMS_PER_SQFT"] = (
            df["BEDROOMS"]
            / df["SQFT"]
            * 1000
        )

        df["LOG_SQFT"] = np.log1p(
            df["SQFT"].clip(lower=0)
        )

        return df

    def preprocess_input(
        self,
        input_data: Dict[str, Any],
    ) -> np.ndarray:
        """Preprocess a single prediction request."""

        df = pd.DataFrame([input_data])

        df = self._prepare_dataframe(df)

        return self.preprocessor.transform(df)

    def preprocess_batch(
        self,
        input_data: pd.DataFrame,
    ) -> np.ndarray:
        """Preprocess a batch of prediction requests."""

        df = self._prepare_dataframe(input_data)

        return self.preprocessor.transform(df)
