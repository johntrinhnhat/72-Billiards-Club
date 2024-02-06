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
df_pool = pd.read_csv('kioviet_pool.csv')

"""
TABLE DATA PROCESS
"""
df_pool['Table_Id'] = df_pool['Table_Id'].astype(int)
df_pool['Date'] = pd.to_datetime(df_pool['Date'])
df_pool['Duration(min)'] = df_pool['Duration(min)'].astype(int)
# Group the data by 'Date' and count the number of occupied tables
df_pool = df_pool.groupby(['Date']).size().reset_index(name='Occupied_Table_Hours')
# Calculate the pool rate by dividing the occupied table hours by the total potential table hours in a day
df_pool['Occupied_Rate(%)'] = ((df_pool['Occupied_Table_Hours'] / (17 * 22)) * 100).round().astype(int)
df_pool['Year'] = df_pool['Date'].dt.year 
df_pool['Month'] = df_pool['Date'].dt.month
df_pool['Day'] = df_pool['Date'].dt.day

"""
SALE DATA PROCESS
"""
# Drop unnecessary feature variable 
df = df.drop(['Customer_Name'], axis=1)
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])

# Aggrerate Dataframe
df_agg = df.groupby('PurchaseDate').agg({'Sales': 'sum', 
                                          'Discount': 'sum', 
                                          'DayOfWeek': 'first'}).reset_index()

vn_holidays = holidays.VN()
df_agg['Is_Holiday'] = df_agg['PurchaseDate'].apply(lambda x: x in vn_holidays)

df_agg['Is_Holiday'] = df_agg['Is_Holiday'].replace({True: 1, False: 0})


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
df_agg['DayOfWeek'] = df_agg['DayOfWeek'].apply(lambda x: day_mapping[x])

df_agg['Year'] = df_agg['PurchaseDate'].dt.year 
df_agg['Month'] = df_agg['PurchaseDate'].dt.month
df_agg['Day'] = df_agg['PurchaseDate'].dt.day


# Define final Ddataframe
df_agg = df_agg[['Year', 'Month', 'Day', 'DayOfWeek', 'Is_Holiday', 'Discount', 'Sales']]
df_pool = df_pool[['Year', 'Month', 'Day', 'Occupied_Rate(%)']]
df_merged = pd.merge(df_agg, df_pool, on=['Year', 'Month', 'Day'])
df_merged = df_merged[['Year', 'Month', 'Day', 'DayOfWeek', 'Occupied_Rate(%)', 'Is_Holiday', 'Discount', 'Sales']]


# print(df, df.dtypes)
print(df_merged)

X = df_merged.drop(['Sales'], axis=1)
y = df_merged['Sales']

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
rf_model = RandomForestRegressor(n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_depth=40, bootstrap=True)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_rf = rf_model.predict(X_test).astype(int)

# Evaluate the random forest regressor model
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
rf_model_score = rf_model.score(X_test, y_test)

# Print out the metrics
print(f"{Fore.BLUE}\nRandom Forest Regressor Model Performance:")
print(f"R^2 Score: {r2_rf}")
print(f"RMSE: {rmse_rf:.2f}")
print(f"MSE: {mse_rf:.2f}")
print(f"MAE: {mae_rf:.2f}")
# print(f"{Fore.RED}\nSales Predict: {y_pred_rf}")
# print(f"{Fore.YELLOW}Sales Actual: {y_test}")
prediction_rf = pd.DataFrame({'Sale_Test': y_test, 'Sale_Predict': y_pred_rf})
print(prediction_rf)
# """
# RandomizedSearchCV
# """

# # Define the parameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300, 400, 500],
#     'max_depth': [None, 10, 20, 30, 40, 50],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }

# # Create a base model
# rf = RandomForestRegressor(random_state=42)

# # Instantiate the random search and fit it like a GridSearchCV
# random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
#                                    n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# # Assuming X_train and y_train are available from your dataset
# random_search.fit(X_train, y_train)

# # The best parameters from fitting the random search:
# best_params = random_search.best_params_
# print("Best parameters:", best_params)

""""""""""""""""" MULTIPLE LINEAR REGRESSION """""""""""""""""
# Defind the model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_lr = lr_model.predict(X_test).astype(int)

# Evaluate the linear regression model
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
lr_model_score = lr_model.score(X_test, y_test)

# Print out the metrics
print(f"{Fore.BLUE}\nMultiple Linear Regression Model Performance:")
print(f"R^2 Score: {r2_lr}")
print(f"RMSE: {rmse_lr:.2f}")
print(f"MSE: {mse_lr:.2f}")
print(f"MAE: {mae_lr:.2f}")
# print(f"{Fore.RED}\nSales Predict: {y_pred_lr}")
# print(f"{Fore.YELLOW}Sales Actual: {y_test}")
prediction_lr = pd.DataFrame({'Sale_Test': y_test, 'Sale_Predict': y_pred_lr})
print(prediction_lr)

# # Visualization
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Sales', alpha=0.5)
# plt.scatter(range(len(y_pred_lr)), y_pred_lr, color='yellow', label='Predicted Sales', alpha=0.5)
# plt.title('Linear Regression Model: Actual vs Predicted Sales')
# plt.xlabel('Data Point Index')
# plt.ylabel('Sales')
# plt.legend()
# plt.show()
