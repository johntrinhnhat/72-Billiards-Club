# Import all of neccessary packages
from colorama import Fore, Back, Style, init
init(autoreset=True)

import holidays
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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

df = pd.get_dummies(df, columns=['DayOfWeek'])

vn_holidays = holidays.VN()
df['Is_Holiday'] = df['PurchaseDate'].apply(lambda x: x in vn_holidays)

# Define final dataframe
df = df[['Year', 'Month', 'Day', 'Hour', 'DayOfWeek_Monday', 'DayOfWeek_Tuesday', 'DayOfWeek_Wednesday', 'DayOfWeek_Thursday', 'DayOfWeek_Friday', 'DayOfWeek_Saturday', 'DayOfWeek_Sunday', 'Is_Holiday', 'Discount', 'Sales']]
print(df)

X = df.drop(['Sales'], axis=1)
y = df['Sales']
# Convert the pandas DataFrame and Series to NumPy arrays
X = X.to_numpy()
y = y.to_numpy()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train Data Shape: {X_train.shape},{y_train.shape}")
print(f"Test Data Shape: {X_test.shape},{y_test.shape}")

""""""""""""""""" RANDOM FOREST REGRESSOR """""""""""""""""
# Defind the model
rf_model = RandomForestRegressor()

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_rf = rf_model.predict(X_test).astype(int)

# Evaluate the rf_model
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
rf_model_score = rf_model.score(X_train, y_train)

# Print out the metrics
print(f"{Fore.BLUE}\nRandom Forest Regressor Performance:")
print(f"The Model score is: {rf_model_score}")
print(f"RMSE: {rmse_rf}")
print(f"MSE: {mse_rf}")
print(f"MAE: {mae_rf}")
print(f"R^2 Score: {r2_rf}")
print(f"{Fore.RED}\nSales Predict: {y_pred_rf}")
print(f"{Fore.YELLOW}Sales Actual: {y_test}")



""""""""""""""""" LINEAR REGRESSION """""""""""""""""
# Define the model
linear_model = LinearRegression()
# Train the model on the training data
linear_model.fit(X_train, y_train)
# Make predictions on the test data
y_pred_lr = linear_model.predict(X_test).astype(int)
# Evaluate the linear_model
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
linear_model_score = linear_model.score(X_train, y_train)
# Print out the metrics
print(f"{Fore.BLUE}\nLinear Regression Performance:")
print(f'The Model score is: {linear_model_score}')
print(f"RMSE: {rmse_lr}")
print(f"MSE: {mse_lr}")
print(f"MAE: {mae_lr}")
print(f"R2 Score: {r2_lr}")

print(f"{Fore.RED}\nSales Predict: {y_pred_lr}")
print(f"{Fore.YELLOW}Sales Actual: {y_test}")
