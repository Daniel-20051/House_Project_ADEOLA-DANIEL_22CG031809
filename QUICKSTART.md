# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Download Dataset

1. Go to https://www.kaggle.com/c/house-prices-advanced-regression-techniques
2. Download the dataset
3. Extract `train.csv` and place it in the `model/` directory

## Step 3: Train the Model

**Option A: Using Python Script**
```bash
cd model
python model_development.py
cd ..
```

**Option B: Using Jupyter Notebook**
```bash
cd model
jupyter notebook model_development.ipynb
# Run all cells
cd ..
```

After training, you should have:
- `model/house_price_model.pkl`
- `model/house_price_model_encoders.pkl`

## Step 4: Run the Web Application

```bash
python app.py
```

Open your browser and go to: http://localhost:5000

## Step 5: Test the Application

1. Fill in the form with house features
2. Click "Predict House Price"
3. View the predicted price

## Deployment

### Render.com

1. Push code to GitHub
2. Create new Web Service on Render
3. Connect GitHub repository
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app`
6. Deploy!

### PythonAnywhere

1. Upload all files via Files tab
2. Open Bash console
3. Install dependencies: `pip3.10 install --user -r requirements.txt`
4. Create web app pointing to `app.py`
5. Reload web app

## Troubleshooting

**Model not found error:**
- Make sure you've trained the model first (Step 3)
- Check that `house_price_model.pkl` exists in `model/` directory

**Import errors:**
- Make sure all dependencies are installed: `pip install -r requirements.txt`

**Port already in use:**
- Change the port in `app.py` or kill the process using port 5000
