import pytest
import pandas as pd
import numpy as np
from src.house_price_prediction.data.ingestion import DataIngestion
from src.house_price_prediction.data.validation import DataValidator
from src.house_price_prediction.data.preprocessing import DataPreprocessor

class TestData:
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'PRICE': [100000, 200000, 300000],
            'SQFT': [1000, 1500, 2000],
            'BEDROOMS': [2, 3, 4],
            'LOCATION': [1, 2, 3],
            'REGION': [1, 2, 3],
            'TITLED': [1, 2, 3],
            'LEASE': [1, 2, 1],
            'FOOTINGS': [1, 2, 3]
        })
    
    def test_data_ingestion(self):
        ingestion = DataIngestion()
        assert ingestion.raw_path.exists() == False  # Won't exist in test
    
    def test_data_validation(self, sample_data):
        validator = DataValidator()
        assert validator.validate_schema(sample_data) == True
        
    def test_data_preprocessing(self, sample_data):
        preprocessor = DataPreprocessor()
        df_mapped = preprocessor.map_categorical(sample_data)
        
        assert 'LOCATION' in df_mapped.columns
        assert df_mapped['LOCATION'].iloc[0] == 'Urban'