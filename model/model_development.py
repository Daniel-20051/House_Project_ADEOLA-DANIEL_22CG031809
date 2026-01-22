"""
House Price Prediction System - Model Development
This script loads, preprocesses, trains, and saves a house price prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

def load_dataset(file_path='train.csv'):
    """
    Load the house prices dataset.
    Note: You need to download the dataset from Kaggle and place it in the model directory.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_path}' not found.")
        print("Please download the 'House Prices: Advanced Regression Techniques' dataset from Kaggle")
        print("and place the train.csv file in the model directory.")
        return None

def preprocess_data(df):
    """
    Perform data preprocessing:
    - Feature selection (6 features from the recommended 9)
    - Handle missing values
    - Encode categorical variables
    - Feature scaling
    """
    # Selected 6 features: OverallQual, GrLivArea, TotalBsmtSF, GarageCars, YearBuilt, Neighborhood
    selected_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt', 'Neighborhood']
    target = 'SalePrice'
    
    # Check if all features exist
    missing_features = [f for f in selected_features + [target] if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        return None, None
    
    # Create a copy with selected features
    data = df[selected_features + [target]].copy()
    
    print(f"\nOriginal data shape: {data.shape}")
    print(f"Selected features: {selected_features}")
    
    # Handle missing values
    print("\nHandling missing values...")
    print(f"Missing values before:\n{data[selected_features].isnull().sum()}")
    
    # Fill missing values
    # For numerical features, use median
    numerical_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt']
    for feature in numerical_features:
        if data[feature].isnull().any():
            data[feature].fillna(data[feature].median(), inplace=True)
    
    # For categorical features, use mode
    categorical_features = ['Neighborhood']
    for feature in categorical_features:
        if data[feature].isnull().any():
            data[feature].fillna(data[feature].mode()[0], inplace=True)
    
    print(f"Missing values after:\n{data[selected_features].isnull().sum()}")
    
    # Encode categorical variables
    print("\nEncoding categorical variables...")
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature].astype(str))
        label_encoders[feature] = le
        print(f"Encoded {feature}: {len(le.classes_)} unique values")
    
    # Separate features and target
    X = data[selected_features]
    y = data[target]
    
    # Remove any remaining rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    print(f"\nFinal data shape: X={X.shape}, y={y.shape}")
    
    return X, y, label_encoders

def train_model(X, y):
    """
    Train a Random Forest Regressor model.
    """
    print("\nTraining Random Forest Regressor...")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Initialize and train the model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate the model
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Training metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    
    print("\nTraining Set Metrics:")
    print(f"  MAE (Mean Absolute Error):  ${train_mae:,.2f}")
    print(f"  MSE (Mean Squared Error):   ${train_mse:,.2f}")
    print(f"  RMSE (Root Mean Squared Error): ${train_rmse:,.2f}")
    print(f"  R² (R-squared):             {train_r2:.4f}")
    
    # Testing metrics
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nTesting Set Metrics:")
    print(f"  MAE (Mean Absolute Error):  ${test_mae:,.2f}")
    print(f"  MSE (Mean Squared Error):   ${test_mse:,.2f}")
    print(f"  RMSE (Root Mean Squared Error): ${test_rmse:,.2f}")
    print(f"  R² (R-squared):             {test_r2:.4f}")
    print("="*50)
    
    return model, X_train, X_test, y_train, y_test

def save_model(model, label_encoders, filepath='model/house_price_model.pkl'):
    """
    Save the trained model and label encoders to disk.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model
    joblib.dump(model, filepath)
    print(f"\nModel saved to: {filepath}")
    
    # Save label encoders
    encoders_path = filepath.replace('.pkl', '_encoders.pkl')
    joblib.dump(label_encoders, encoders_path)
    print(f"Label encoders saved to: {encoders_path}")
    
    return filepath, encoders_path

def main():
    """
    Main function to run the complete model development pipeline.
    """
    print("="*50)
    print("HOUSE PRICE PREDICTION SYSTEM - MODEL DEVELOPMENT")
    print("="*50)
    
    # Load dataset
    # Try different possible locations
    dataset_paths = [
        'train.csv',
        'model/train.csv',
        '../train.csv',
        'data/train.csv'
    ]
    
    df = None
    for path in dataset_paths:
        if os.path.exists(path):
            df = load_dataset(path)
            break
    
    if df is None:
        print("\nPlease ensure the dataset file (train.csv) is available.")
        print("You can download it from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques")
        return
    
    # Preprocess data
    X, y, label_encoders = preprocess_data(df)
    
    if X is None:
        print("Error in data preprocessing. Exiting.")
        return
    
    # Train model
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # Save model
    model_path, encoders_path = save_model(model, label_encoders)
    
    print("\n" + "="*50)
    print("MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"\nModel saved at: {model_path}")
    print(f"Encoders saved at: {encoders_path}")
    print("\nYou can now use the saved model in the web application.")

if __name__ == "__main__":
    main()
