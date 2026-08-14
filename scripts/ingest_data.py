#!/usr/bin/env python
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.house_price_prediction.data.ingestion import DataIngestion
from src.house_price_prediction.data.validation import DataValidator
from src.house_price_prediction.data.preprocessing import DataPreprocessor
from src.house_price_prediction.utils.logger import logger

def main():
    logger.info("Starting data ingestion process...")
    
    try:
        # Initialize components
        ingestion = DataIngestion()
        validator = DataValidator()
        preprocessor = DataPreprocessor()

        # Load raw data
        df = ingestion.load_data()

        # Validate data
        if not validator.validate_schema(df):
            raise ValueError("Data schema validation failed")
        
        quality_report = validator.validate_data_quality(df)
        logger.info(f"Data quality report: {quality_report}")
        
        # Clean data
        df_clean = validator.clean_data(df)
        
        # Map categorical variables
        df_mapped = preprocessor.map_categorical(df_clean)
        
        # Engineer features
        df_engineered = preprocessor.engineer_features(df_mapped)
        
        # Save processed data
        ingestion.save_processed_data(df_engineered)
        
        logger.info("Data ingestion and preprocessing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data ingestion: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
