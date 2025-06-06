import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import re

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

def force_numeric(s):
    """Convert series to numeric, extracting numbers from strings if needed"""
    return pd.to_numeric(s.astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce')

def load_data(file_path):
    """Load and preprocess data"""
    df = pd.read_csv(file_path)
    print(f"Data loaded. Shape: {df.shape}\nOriginal dtypes:\n{df.dtypes}")

    # Column definitions
    num_cols = ['Amount(in rupees)', 'Price (in rupees)', 'Carpet Area', 
               'Bathroom', 'Balcony', 'Car Parking', 'Super Area']
    cat_cols = ['location', 'Status', 'Transaction', 'Furnishing', 
               'facing', 'overlooking', 'Ownership']

    # Clean numeric columns
    print("\nConverting numeric columns:")
    for col in num_cols:
        if col in df:
            df[col] = force_numeric(df[col])
            median = df[col].median()
            df[col].fillna(median, inplace=True)
            print(f"{col}: dtype={df[col].dtype}, missing filled with {median:.2f}")

    # Clean categorical columns
    print("\nFilling categorical missing values:")
    for col in cat_cols:
        if col in df:
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)
            print(f"{col}: filled with '{mode}'")

    return df

def preprocess_data(df):
    """Feature engineering and encoding"""
    cat_cols = ['location', 'Status', 'Transaction', 'Furnishing', 
               'facing', 'overlooking', 'Ownership']
    df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df])
    
    features = ['Amount(in rupees)', 'Carpet Area', 'Super Area', 
               'Bathroom', 'Balcony', 'Car Parking'] + \
              [c for c in df.columns if any(c.startswith(x) for x in cat_cols)]
    
    print(f"\nSelected {len(features)} features. First 10: {features[:10]}")
    return df[[c for c in features if c in df]], df['Price (in rupees)']

def train_model(X, y):
    """Train and evaluate model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression().fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation metrics
    mse, rmse, r2 = mean_squared_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)
    print("\n" + "="*50 + f"\nLinear Regression Results:\nMSE: ₹{mse:,.2f}\nRMSE: ₹{rmse:,.2f}\nR²: {r2:.4f}\n" + "="*50)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price (₹)'); plt.ylabel('Predicted Price (₹)')
    plt.title('Actual vs Predicted House Prices\nLinear Regression', pad=20)
    plt.grid(True)
    plt.text(0.05, 0.9, f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}\nR² = {r2:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    plot_path = os.path.join('plots', 'linear_regression_results.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved plot: '{plot_path}'")
    
    return model, scaler

if __name__ == "__main__":
    print("\n" + "="*50 + "\nStarting House Price Prediction Pipeline\n" + "="*50)
    try:
        df = load_data('data/house_data.csv')
        X, y = preprocess_data(df)
        model, scaler = train_model(X, y)
        
        joblib.dump(model, 'models/linear_regression_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        print("\nSaved models:\n- models/linear_regression_model.pkl\n- models/scaler.pkl\n\nPipeline completed!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")