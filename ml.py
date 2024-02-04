# Import all of neccessary packages
from colorama import Fore, Back, Style, init
init(autoreset=True)
from xgboost import XGBRegressor
import requests
import json
import holidays
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Load dataframe for machine learning project
df = pd.read_csv('kioviet.csv')
# Drop unnecessary feature variable 
df = df.drop(['Customer_Name'], axis=1)

df['Sales'] = df['Sales'].astype(int)

# Convert 'PurchaseDate' to datetime object
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])

# Extract features variables from 'PurchaseDate'
df['Year'] = df['PurchaseDate'].dt.year 
df['Month'] = df['PurchaseDate'].dt.month
df['Day'] = df['PurchaseDate'].dt.day
df['Hour'] = df['PurchaseDate'].dt.hour


vn_holidays = holidays.VN()
df['Is_Holiday'] = df['PurchaseDate'].apply(lambda x: x in vn_holidays)
df['Is_Holiday'] = df['Is_Holiday'].replace({True: 1, False: 0})

# Define a function that returns the rate based on the hour
def get_table_rate(hour):
    if 1 <= hour <= 6:
        return 85000
    elif 9 <= hour <= 12:
        return 55000
    elif 13 <= hour <= 17:
        return 75000
    elif hour == 0 or 18 <= hour <= 23:
        return 95000
    else:
        # If the hour doesn't fall into any of the specified ranges,
        # you can define a default rate or raise an error.
        return None
    
# Apply this function to the 'Hour' column to create a new 'TableRate' column
df['Table_Rate'] = df['Hour'].apply(get_table_rate)

# Mapping from day name to numerical value where Monday is 0 and Sunday is 6
day_mapping = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}
# Apply this mapping to the 'DayOfWeek' column
df['DayOfWeek'] = df['DayOfWeek'].apply(lambda x: day_mapping[x])

# Define final dataframe
df = df[['Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'Table_Rate', 'Is_Holiday', 'Discount', 'Sales']]
print(df, df.dtypes)

X = df.drop(['Sales'], axis=1)
y = df['Sales']

feature_names = X.columns

# Convert the pandas DataFrame and Series to NumPy arrays
X = X.to_numpy()
y = y.to_numpy()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f"Train Data Shape: {X_train.shape},{y_train.shape}")
print(f"Test Data Shape: {X_test.shape},{y_test.shape}")

""""""""""""""""" RANDOM FOREST REGRESSOR """""""""""""""""
# Defind the model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_rf = rf_model.predict(X_test).astype(int)

# Evaluate the rf_model
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))/1000
mse_rf = mean_squared_error(y_test, y_pred_rf)/1000000
mae_rf = mean_absolute_error(y_test, y_pred_rf)/1000
r2_rf = r2_score(y_test, y_pred_rf)
rf_model_score = rf_model.score(X_test, y_test)

# Print out the metrics
print(f"{Fore.BLUE}\nRandom Forest Regressor Performance:")
print(f"The Model score is: {rf_model_score}")
print(f"RMSE: {rmse_rf:.2f}k đ")
print(f"MSE: {mse_rf:.2f}M đ")
print(f"MAE: {mae_rf:.2f}k đ")
print(f"R^2 Score: {r2_rf}")
print(f"{Fore.RED}\nSales Predict: {y_pred_rf}")
print(f"{Fore.YELLOW}Sales Actual: {y_test}")



# """"""""""""""""" LINEAR REGRESSION """""""""""""""""
# # Define the model
# linear_model = LinearRegression()
# # Train the model on the training data
# linear_model.fit(X_train, y_train)
# # Make predictions on the test data
# y_pred_lr = linear_model.predict(X_test).astype(int)
# # Evaluate the linear_model
# rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))/1000
# mse_lr = mean_squared_error(y_test, y_pred_lr)/1000000
# mae_lr = mean_absolute_error(y_test, y_pred_lr)/1000
# r2_lr = r2_score(y_test, y_pred_lr)
# linear_model_score = linear_model.score(X_test, y_test)
# # Print out the metrics
# print(f"{Fore.BLUE}\nLinear Regression Performance:")
# print(f'The Model score is: {linear_model_score}')
# print(f"RMSE: {rmse_lr:.2f}k đ")
# print(f"MSE: {mse_lr:.2f}M đ")
# print(f"MAE: {mae_lr:.2f}k đ")
# print(f"R2 Score: {r2_lr}")

# print(f"{Fore.RED}\nSales Predict: {y_pred_lr}")
# print(f"{Fore.YELLOW}Sales Actual: {y_test}")


# correlation_matrix = df.corr()
# sales_correlation = correlation_matrix['Sales'].sort_values(ascending=False)
# print(f"Correlation: \n{sales_correlation}")

# # Assuming `rf_model` is a fitted RandomForestRegressor
# importances = rf_model.feature_importances_
# feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# print(f"Important Features: \n{feature_importances}")