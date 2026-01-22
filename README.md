# House Price Prediction System

A machine learning web application that predicts house prices based on key features from the House Prices: Advanced Regression Techniques dataset.

## Project Structure

```
HousePrice_Project_yourName_matricNo/
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── HousePrice_hosted_webGUI_link.txt  # Deployment information
├── model/
│   ├── model_development.py        # Model training script
│   ├── model_development.ipynb     # Model training notebook
│   └── house_price_model.pkl      # Trained model (generated after training)
├── static/
│   └── style.css                   # CSS stylesheet
└── templates/
    └── index.html                  # Web interface
```

## Features

- **Machine Learning Model**: Random Forest Regressor trained on 6 selected features
- **Web Interface**: User-friendly Flask web application
- **Real-time Prediction**: Instant house price predictions based on user input
- **Model Persistence**: Saved model using Joblib for easy loading

## Selected Features

The model uses the following 6 features:
1. **OverallQual**: Overall quality rating (1-10)
2. **GrLivArea**: Above grade living area (square feet)
3. **TotalBsmtSF**: Total basement area (square feet)
4. **GarageCars**: Garage capacity (number of cars)
5. **YearBuilt**: Original construction year
6. **Neighborhood**: Physical location within Ames city limits

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the "House Prices: Advanced Regression Techniques" dataset from Kaggle:
- URL: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
- Place the `train.csv` file in the `model/` directory

### 3. Train the Model

**Option A: Using Python Script**
```bash
cd model
python model_development.py
```

**Option B: Using Jupyter Notebook**
```bash
cd model
jupyter notebook model_development.ipynb
```

This will generate:
- `house_price_model.pkl`: The trained model
- `house_price_model_encoders.pkl`: Label encoders for categorical variables

### 4. Run the Web Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Model Evaluation Metrics

The model is evaluated using:
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (R-squared)

## Deployment

### Using Render.com

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn app:app`
5. Deploy!

### Using PythonAnywhere

1. Upload your project files
2. Create a new web app
3. Point it to your `app.py` file
4. Reload the web app

### Using Streamlit Cloud

If you prefer Streamlit, you can convert the Flask app to Streamlit format.

## Technologies Used

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn (Random Forest Regressor)
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib
- **Frontend**: HTML, CSS, JavaScript

## License

This project is created for educational purposes.

## Author

[Your Name]
[Your Matric Number]
