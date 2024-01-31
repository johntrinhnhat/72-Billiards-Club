# Import all of neccessary packages
import holidays
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error


# Load dataframe for machine learning project
df = pd.read_csv('kioviet.csv')
# Drop unnecessary feature variable 
df = df.drop(['Customer_Name'], axis=1)

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
df = df[['Year', 'Month', 'Day', 'Hour', 'DayOfWeek_Monday', 'DayOfWeek_Tuesday', 'DayOfWeek_Wednesday', 'DayOfWeek_Thursday', 'DayOfWeek_Friday', 'DayOfWeek_Saturday', 'DayOfWeek_Sunday', 'Is_Holiday','Sales']]
# print(df[(df['Is_Holiday'] == True)])

X = df.drop(['Sales'], axis=1)
y = df['Sales']
# print(df.isnull().sum())
# Convert the pandas DataFrame and Series to NumPy arrays
X = X.to_numpy()
y = y.to_numpy()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

""""""""""""""""" RANDOM FOREST REGRESSOR """""""""""""""""
# Defind the model
rf_model = RandomForestRegressor()
# Train the model on the training data
rf_model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = rf_model.predict(X_test)
# Evaluate the rf_model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
rf_model_score = rf_model.score(X_train, y_train)
# Print out the metrics
print("\nRandom Forest Regressor Performance:")
print(f"The Model score is: {rf_model_score}")
print(f"RMSE: {rmse}")
print(f"R^2 Score: {r2}")


""""""""""""""""" LINEAR REGRESSION """""""""""""""""
linear_model = LinearRegression()
# Train the model on the training data
linear_model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = linear_model.predict(X_test)
# linear_predictions = linear_model.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
# Evaluate the linear_model
print("\nLinear Regression Performance:")
print('The model score is:', linear_model.score(X_train, y_train))
print("RMSE:", rmse)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

print(f"\nSales Predict:",y_pred)
print(f"Sales Actual:",y_test)