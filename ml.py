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

"""
SALE DATA PROCESS
"""
# Drop unnecessary feature variable 
df = df.drop(['Customer_Name'], axis=1)
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
df['Occupied_Table_Hours'] = df['Duration(min)'] / 60
# Aggrerate Dataframe
df_agg = df.groupby('PurchaseDate').agg({'Sales': 'sum', 
                                          'Discount': 'sum', 
                                          'DayOfWeek': 'first',
                                          'Occupied_Table_Hours': 'size'}).reset_index()

df_agg['Occupied_Rate(%)'] = ((df_agg['Occupied_Table_Hours'] / (17 * 20)) * 100).round().astype(int)
vn_holidays = holidays.VN()
df_agg['Is_Holiday'] = df_agg['PurchaseDate'].apply(lambda x: x in vn_holidays)

df_agg['Is_Holiday'] = df_agg['Is_Holiday'].replace({True: 1, False: 0})

# Apply this mapping to the 'DayOfWeek' column
df_agg['Year'] = df_agg['PurchaseDate'].dt.year 
df_agg['Month'] = df_agg['PurchaseDate'].dt.month
df_agg['Day'] = df_agg['PurchaseDate'].dt.day
df_agg['DayOfWeek'] = df_agg['PurchaseDate'].dt.dayofweek


# Define final Ddataframe
df_agg = df_agg[['Year', 'Month', 'Day', 'DayOfWeek', 'Occupied_Rate(%)', 'Is_Holiday', 'Discount', 'Sales']]


print(df_agg, df_agg.dtypes)

X = df_agg.drop(['Sales'], axis=1)
y = df_agg['Sales']

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
prediction_rf = pd.DataFrame({'Sale_Test': y_test, 'Sale_Predict': y_pred_rf, 'Difference': y_test-y_pred_rf})
print(prediction_rf)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Sales', alpha=0.5)
plt.scatter(range(len(y_pred_rf)), y_pred_rf, color='yellow', label='Predicted Sales', alpha=0.5)
plt.title('Random Forest Regressor Model: Actual vs Predicted Sales')
plt.xlabel('Data Point Index')
plt.ylabel('Sales')
plt.legend()
plt.show()


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
prediction_lr = pd.DataFrame({'Sale_Test': y_test, 'Sale_Predict': y_pred_lr, 'Difference': y_test-y_pred_lr})
print(prediction_lr)

# Visualization
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Sales', alpha=0.5)
# plt.scatter(range(len(y_pred_lr)), y_pred_lr, color='yellow', label='Predicted Sales', alpha=0.5)
# plt.title('Multiple Linear Regression Model: Actual vs Predicted Sales')
# plt.xlabel('Data Point Index')
# plt.ylabel('Sales')~
# plt.legend()
# plt.show()
