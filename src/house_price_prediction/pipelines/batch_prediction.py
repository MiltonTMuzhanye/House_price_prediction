"""
Batch prediction pipeline for processing multiple inputs efficiently.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

from .inference_pipeline import InferencePipeline
from ..utils.logger import logger
from ..utils.config import config
from ..utils.helpers import save_artifact

class BatchPredictor:
    """Handles batch predictions with various input formats."""
    
    def __init__(self):
        self.inference_pipeline = InferencePipeline()
        self.batch_results = []
    
    def predict_from_csv(self, csv_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """Make predictions from CSV file."""
        logger.info(f"Reading CSV from: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} rows from CSV")
            
            # Make predictions
            results = self.inference_pipeline.predict_dataframe(df)
            
            # Save if output path provided
            if output_path:
                results.to_csv(output_path, index=False)
                logger.info(f"Results saved to: {output_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise
    
    def predict_from_json(self, json_path: str, output_path: Optional[str] = None) -> List[Dict]:
        """Make predictions from JSON file."""
        logger.info(f"Reading JSON from: {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                features_list = data
            elif isinstance(data, dict) and 'features' in data:
                features_list = data['features']
            else:
                features_list = [data]
            
            logger.info(f"Loaded {len(features_list)} samples from JSON")
            
            # Make predictions
            results = self.inference_pipeline.predict_batch(features_list)
            
            # Save if output path provided
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to: {output_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing JSON: {str(e)}")
            raise
    
    def predict_parallel(self, features_list: List[Dict], max_workers: int = 4) -> List[Dict]:
        """Make predictions in parallel for better performance."""
        logger.info(f"Making parallel predictions with {max_workers} workers...")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_features = {
                executor.submit(self.inference_pipeline.predict_single, features): features
                for features in features_list
            }
            
            for future in as_completed(future_to_features):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Parallel prediction error: {str(e)}")
                    results.append({'error': str(e)})
        
        logger.info(f"Completed {len(results)} parallel predictions")
        return results
    
    def predict_with_progress(self, features_list: List[Dict], batch_size: int = 100) -> List[Dict]:
        """Make predictions with progress tracking."""
        logger.info(f"Making predictions for {len(features_list)} samples with batch size {batch_size}...")
        
        results = []
        total_batches = (len(features_list) + batch_size - 1) // batch_size
        
        for i in range(0, len(features_list), batch_size):
            batch = features_list[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} samples)...")
            
            batch_results = self.inference_pipeline.predict_batch(batch)
            results.extend(batch_results)
            
            # Log progress
            progress = (i + len(batch)) / len(features_list) * 100
            logger.info(f"Progress: {progress:.1f}%")
        
        return results
    
    def export_results(self, results: List[Dict], format: str = 'json', filepath: Optional[str] = None):
        """Export batch results in various formats."""
        if filepath is None:
            filepath = f"reports/batch_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
        elif format == 'csv':
            df = pd.DataFrame(results)
            df.to_csv(filepath, index=False)
        elif format == 'excel':
            df = pd.DataFrame(results)
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results exported to: {filepath}")
        return filepath

# Convenience function
def batch_predict(input_path: str, output_path: Optional[str] = None):
    """Run batch prediction from file."""
    predictor = BatchPredictor()
    
    if input_path.endswith('.csv'):
        return predictor.predict_from_csv(input_path, output_path)
    elif input_path.endswith('.json'):
        return predictor.predict_from_json(input_path, output_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")