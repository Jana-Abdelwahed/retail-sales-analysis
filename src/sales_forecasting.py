import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def train_sales_forecast_model(file_path="../Data_set/cleaned_retail_sales.csv"):
    """
    Aggregates monthly sales and trains a Random Forest model using lag features
    to forecast future retail revenue.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        file_path = "Data_set/cleaned_retail_sales.csv"
        df = pd.read_csv(file_path)
        
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    
    monthly_sales = (
        df.groupby(df['OrderDate'].dt.to_period('M'))['Calculated_Revenue']
        .sum()
        .reset_index()
    )
    
    monthly_sales['OrderDate'] = monthly_sales['OrderDate'].dt.to_timestamp()
    monthly_sales = monthly_sales.sort_values('OrderDate').reset_index(drop=True)
    
    monthly_sales['Lag_1_Month'] = monthly_sales['Calculated_Revenue'].shift(1)
    monthly_sales['Lag_2_Month'] = monthly_sales['Calculated_Revenue'].shift(2)
    monthly_sales['Rolling_Mean_3'] = monthly_sales['Calculated_Revenue'].shift(1).rolling(window=3).mean()
    
    monthly_sales = monthly_sales.dropna().reset_index(drop=True)
    
    if len(monthly_sales) < 6:
        print("Warning: Not enough monthly data points for robust training. Consider using a larger dataset.")
        return monthly_sales

    X = monthly_sales[['Lag_1_Month', 'Lag_2_Month', 'Rolling_Mean_3']]
    y = monthly_sales['Calculated_Revenue']
    
    train_size = int(len(monthly_sales) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Sales Forecasting Model - Mean Absolute Error: ${mae:,.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales['OrderDate'], monthly_sales['Calculated_Revenue'], label='Actual Revenue', marker='o', color='blue')
    
    test_dates = monthly_sales['OrderDate'].iloc[train_size:]
    plt.plot(test_dates, predictions, label='Predicted Revenue', marker='x', linestyle='--', color='orange')
    
    plt.title('Retail Sales Forecasting (Random Forest with Lag Features)', fontsize=13, fontweight='bold')
    plt.xlabel('Timeline', fontsize=10)
    plt.ylabel('Revenue ($)', fontsize=10)
    plt.legend()
    plt.savefig('sales_forecast.png', dpi=300, bbox_inches='tight')
    print("Sales forecast plot saved to 'sales_forecast.png'")
    plt.show()
    
    return model

if __name__ == "__main__":
    train_sales_forecast_model()