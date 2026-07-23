import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from .logger import logger

def save_artifact(obj: Any, path: str):
    """Save artifact using joblib"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info(f"Saved artifact to {path}")

def load_artifact(path: str) -> Any:
    """Load artifact using joblib"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)

def calculate_price_per_sqft(df: pd.DataFrame, price_col: str = "PRICE", sqft_col: str = "SQFT") -> pd.Series:
    """Calculate price per square foot"""
    return df[price_col] / df[sqft_col]

def create_directory_structure(base_path: str = "."):
    """Create all required directories"""
    directories = [
        "data/raw",
        "data/processed",
        "data/external",
        "artifacts/trained_models",
        "artifacts/scalers",
        "artifacts/encoders",
        "artifacts/feature_lists",
        "reports/figures",
        "reports/metrics",
        "reports/model_cards",
        "experiments/notebooks",
        "experiments/mlruns",
        "logs",
        "mlflow",
        "monitoring/logging",
        "monitoring/drift",
        "deployment/docker",
        "deployment/kubernetes",
        "deployment/terraform"
    ]
    
    for directory in directories:
        path = Path(base_path) / directory
        path.mkdir(parents=True, exist_ok=True)

def get_feature_importance(model, feature_names):
    """Extract feature importance from trained model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        raise ValueError("Model does not have feature_importances_ or coef_")
    
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)