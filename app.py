"""
House Price Prediction System - Web Application
Flask web application for predicting house prices using the trained model.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and encoders
MODEL_PATH = 'model/house_price_model.pkl'
ENCODERS_PATH = 'model/house_price_model_encoders.pkl'

model = None
label_encoders = None

def load_model():
    """Load the trained model and label encoders."""
    global model, label_encoders
    
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
            print("Please run model_development.py or model_development.ipynb first to train the model.")
            return False
        
        if os.path.exists(ENCODERS_PATH):
            label_encoders = joblib.load(ENCODERS_PATH)
            print(f"Label encoders loaded successfully from {ENCODERS_PATH}")
        else:
            print(f"Warning: Encoders file not found at {ENCODERS_PATH}")
            label_encoders = {}
        
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Load model on startup
load_model()

# Neighborhood options (common neighborhoods from the dataset)
# These should match the neighborhoods used during training
NEIGHBORHOODS = [
    'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr',
    'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
    'NAmes', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown',
    'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber',
    'Veenker'
]

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', neighborhoods=NEIGHBORHOODS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure the model file exists.'
            }), 500
        
        # Get input data from the form
        data = request.get_json()
        
        # Extract features
        overall_qual = float(data.get('OverallQual', 5))
        gr_liv_area = float(data.get('GrLivArea', 1500))
        total_bsmt_sf = float(data.get('TotalBsmtSF', 1000))
        garage_cars = float(data.get('GarageCars', 2))
        year_built = float(data.get('YearBuilt', 2000))
        neighborhood = data.get('Neighborhood', 'NAmes')
        
        # Encode neighborhood
        neighborhood_encoded = neighborhood
        if 'Neighborhood' in label_encoders:
            try:
                neighborhood_encoded = label_encoders['Neighborhood'].transform([neighborhood])[0]
            except (ValueError, KeyError):
                # If neighborhood not in training data, use a default value
                # Use the most common neighborhood encoding (usually 0 or middle value)
                neighborhood_encoded = 0
        
        # Prepare feature array in the correct order
        features = np.array([[
            overall_qual,
            gr_liv_area,
            total_bsmt_sf,
            garage_cars,
            year_built,
            neighborhood_encoded
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Ensure prediction is positive
        prediction = max(0, prediction)
        
        return jsonify({
            'predicted_price': float(prediction),
            'formatted_price': f'${prediction:,.2f}'
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # For development
    app.run(debug=True, host='0.0.0.0', port=5000)
    # For production, use: app.run(host='0.0.0.0', port=5000)
