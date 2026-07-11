#!/usr/bin/env python
import sys
from pathlib import Path
import json
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.inference.predictor import HousePricePredictor
from src.house_price_prediction.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="Make house price predictions")
    parser.add_argument("--input", type=str, help="JSON input file or JSON string")
    parser.add_argument("--batch", action="store_true", help="Batch prediction mode")
    args = parser.parse_args()
    
    try:
        predictor = HousePricePredictor()
        
        if args.input:
            # Parse input
            try:
                with open(args.input, 'r') as f:
                    input_data = json.load(f)
            except FileNotFoundError:
                input_data = json.loads(args.input)
            
            if args.batch:
                if not isinstance(input_data, list):
                    raise ValueError("Batch input should be a list")
                results = predictor.predict_batch(input_data)
                print(json.dumps(results, indent=2))
            else:
                result = predictor.predict_single(input_data)
                print(json.dumps(result, indent=2))
        else:
            # Interactive mode
            logger.info("Interactive prediction mode")
            logger.info("Enter house features:")
            
            features = {
                "SQFT": int(input("Square Footage (100-5000): ")),
                "BEDROOMS": int(input("Number of Bedrooms (1-10): ")),
                "LOCATION": int(input("Location (1:Urban, 2:Suburban, 3:Rural): ")),
                "REGION": int(input("Region (1:Northeast, 2:Midwest, 3:South, 4:West): ")),
                "TITLED": int(input("Title (1:Vehicle, 2:Land-Home, 3:Other): ")),
                "LEASE": int(input("Lease (0:No, 1:Yes): ")),
                "FOOTINGS": int(input("Footing Type (1-9): "))
            }
            
            result = predictor.predict_single(features)
            print(f"\nPredicted Price: {result['predicted_price_formatted']}")
            
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()