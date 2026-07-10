import pandas as pd
from pathlib import Path
from typing import Optional
from ..utils.logger import logger
from ..utils.config import config

class DataIngestion:
    """Handles data ingestion from various sources"""
    
    def __init__(self):
        self.raw_path = Path(config.get('data.raw_path', 'data/raw/PUF2023.xlsx'))
        self.processed_path = Path(config.get('data.processed_path', 'data/processed/house_prices_processed.csv'))
        
    def load_data(self) -> pd.DataFrame:
        """Load data from Excel file"""
        try:
            if not self.raw_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.raw_path}")
            
            df = pd.read_excel(self.raw_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def save_processed_data(self, df: pd.DataFrame):
        """Save processed data to CSV"""
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.processed_path, index=False)
        logger.info(f"Saved processed data to {self.processed_path}")