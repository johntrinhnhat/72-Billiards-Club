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
# # Extract features variables from 'PurchaseDate'
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
# df['Year'] = df['PurchaseDate'].dt.year 
# df['Month'] = df['PurchaseDate'].dt.month
# df['Day'] = df['PurchaseDate'].dt.day
df['Hour'] = df['PurchaseDate'].dt.hour

# Aggrerate Dataframe
df_copy = df.groupby('PurchaseDate').agg({'Sales': 'sum', 
                                          'Discount': 'sum', 
                                          'DayOfWeek': 'first'}).reset_index()

vn_holidays = holidays.VN()
df_copy['Is_Holiday'] = df_copy['PurchaseDate'].apply(lambda x: x in vn_holidays)

df_copy['Is_Holiday'] = df_copy['Is_Holiday'].replace({True: 1, False: 0})


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
df_copy['DayOfWeek'] = df_copy['DayOfWeek'].apply(lambda x: day_mapping[x])

df_copy['Year'] = df_copy['PurchaseDate'].dt.year 
df_copy['Month'] = df_copy['PurchaseDate'].dt.month
df_copy['Day'] = df_copy['PurchaseDate'].dt.day

# Define final Ddataframe
df_copy = df_copy[['Year', 'Month', 'Day', 'DayOfWeek', 'Is_Holiday', 'Discount', 'Sales']]
# print(df, df.dtypes)
print(df_copy, df_copy.dtypes)

X = df_copy.drop(['Sales'], axis=1)
y = df_copy['Sales']

# Convert the pandas DataFrame and Series to NumPy arrays
X = X.to_numpy()
y = y.to_numpy()

# # Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData Shape:")
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


