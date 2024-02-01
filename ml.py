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

df = pd.get_dummies(df, columns=['DayOfWeek'])

vn_holidays = holidays.VN()
df['Is_Holiday'] = df['PurchaseDate'].apply(lambda x: x in vn_holidays)

# Define final dataframe
df = df[['PurchaseDate', 'Year', 'Month', 'Day', 'Hour', 'DayOfWeek_Monday', 'DayOfWeek_Tuesday', 'DayOfWeek_Wednesday', 'DayOfWeek_Thursday', 'DayOfWeek_Friday', 'DayOfWeek_Saturday', 'DayOfWeek_Sunday', 'Is_Holiday', 'Discount', 'Sales']]
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
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_rf = rf_model.predict(X_test).astype(int)

# Evaluate the rf_model
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))/1000
mse_rf = mean_squared_error(y_test, y_pred_rf)/1000000
mae_rf = mean_absolute_error(y_test, y_pred_rf)/1000
r2_rf = r2_score(y_test, y_pred_rf)
rf_model_score = rf_model.score(X_train, y_train)

# Print out the metrics
print(f"{Fore.BLUE}\nRandom Forest Regressor Performance:")
print(f"The Model score is: {rf_model_score}")
print(f"RMSE: {rmse_rf:.2f}k đ")
print(f"MSE: {mse_rf:.2f}M đ")
print(f"MAE: {mae_rf:.2f}k đ")
print(f"R^2 Score: {r2_rf}")
print(f"{Fore.RED}\nSales Predict: {y_pred_rf}")
print(f"{Fore.YELLOW}Sales Actual: {y_test}")



""""""""""""""""" LINEAR REGRESSION """""""""""""""""
# Define the model
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
# linear_model_score = linear_model.score(X_train, y_train)
# Print out the metrics
# print(f"{Fore.BLUE}\nLinear Regression Performance:")
# print(f'The Model score is: {linear_model_score}')
# print(f"RMSE: {rmse_lr:.2f}k đ")
# print(f"MSE: {mse_lr:.2f}M đ")
# print(f"MAE: {mae_lr:.2f}k đ")
# print(f"R2 Score: {r2_lr}")

# print(f"{Fore.RED}\nSales Predict: {y_pred_lr}")
# print(f"{Fore.YELLOW}Sales Actual: {y_test}")


# Replace 'your_api_key' with your actual API key
api_key = '17a2af3a1cf36f4fe30ea1f9a07181d9'
city_name = 'Ho Chi Minh'

# Build the API URL
url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"

# Make a GET request to fetch the weather data
response = requests.get(url)
weather_data = response.json()

# Check the response status code
if response.status_code == 200:
    # If the request was successful, parse the data
    temperature = weather_data['main']['temp']
    
    # Print the weather information
    print(f"Temperature: {temperature}")
else:
    # If the request failed, print an error message
    print("Error fetching data from the OpenWeatherMap API.")
