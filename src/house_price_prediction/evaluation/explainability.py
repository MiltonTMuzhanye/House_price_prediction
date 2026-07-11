import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import shap
from ..utils.logger import logger
from ..utils.helpers import save_artifact

class ModelExplainer:
    """Provides model explainability and interpretation"""
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def create_shap_explainer(self, X_sample):
        """Create SHAP explainer"""
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(X_sample)
            logger.info("SHAP explainer created successfully")
            return self.shap_values
        except Exception as e:
            logger.error(f"Error creating SHAP explainer: {str(e)}")
            return None
    
    def plot_shap_summary(self, X_sample, save_path=None):
        """Plot SHAP summary"""
        if self.shap_values is None:
            self.create_shap_explainer(X_sample)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X_sample, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
        
        plt.show()
    
    def get_global_importance(self) -> pd.DataFrame:
        """Get global feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            raise ValueError("Model doesn't support feature importance")
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def partial_dependence_plots(self, X, feature, save_path=None):
        """Create partial dependence plot for a feature"""
        # Simplified version - actual implementation would use PDP from sklearn
        feature_idx = self.feature_names.index(feature)
        
        plt.figure(figsize=(10, 6))
        # Create grid of values
        grid = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), 50)
        predictions = []
        
        for val in grid:
            X_temp = X.copy()
            X_temp[:, feature_idx] = val
            predictions.append(np.mean(self.model.predict(X_temp)))
        
        plt.plot(grid, predictions, 'b-', linewidth=2)
        plt.xlabel(feature)
        plt.ylabel('Partial Dependence')
        plt.title(f'Partial Dependence Plot: {feature}')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PDP plot saved to {save_path}")
        
        plt.show()