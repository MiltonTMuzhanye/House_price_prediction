# House_price_prediction

This project focuses on predicting house prices using machine learning regression techniques. The dataset includes features like as location, number of bedrooms, size in square feet and many more. It's designed to show the full data science lifecycle data preprocessing, feature engineering, model evaluation, and hyperparameter tuning


## Dataset

  The dataset used includes housing information such as:
    -Location
    -Number of bedrooms, bathrooms
    -Total square footage (SQFT)
    -Price per square foot (PRICE_PER_SQFT)
    -Target: Price



### 1. Data Cleaning & Preprocessing
 - remove outliers using inter quartile range
 - created a new feature called "PRICE_PER_SQFT"
 - label encoded cartegorical data

### 2. Model Training

Three models were trained and evaluated:
  -Linear regression 
  -Random forest regressor
  -XGBoost regressor

### Evalueated metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R^2 Score

### 3. Best Model Performance

  # LinearRegression Performance:
    -MAE: 7091.70
    -RMSE: 12472.95
    -R^2 Score: 0.942
    
  # RandomForest Performance:
    -MAE: 1172.61
    -RMSE: 6272.54
    -R^2 Score: 0.985
    
  # XGBoost Performance:
    MAE: 926.51
    RMSE: 4601.07
    R^2 Score: 0.992

### 4. Hyperparameter Tuning (XGBoost)
  # Best parameters:
    - learning_rate: 0.1
    - n_estimators: 200
    - max_depth: 5

    Best R^2 Score: 0.991
    
### 5. Cross Validation (Random Forest)
  Mean CV R^2: 0.995
  
### 6. Feature Importance
  # Top features:
  - SQFT (importance > 0.5)
  - PRICE_PER_SQFT (importance is between 0.4 - 0.5)

## 📈 Visualization
  - Distribution of price
  - Correlation heatmaps
  - Feature importance bar chart
  
## 💡 Conclusion
  XGBoost Regressor gave the best results with high R^2 and low error values. Feature engineering and hyperparameter tuning boosted the performance

-📧 Contact: muzhanyemilton@gmail.com
-🔗 LinkedIn: https://www.linkedin.com/in/milton-muzhanye-900a01348
-📦 GitHub: https://github.com/MiltonTMuzhanye
