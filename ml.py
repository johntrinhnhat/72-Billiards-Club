# Import all of neccessary packages
import holidays
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model on the training data
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = rf_model.predict(X_test)
# Evaluate the rf_model
rf_model_score = rf_model.score(X_train, y_train)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
# Print out the metrics
print(f"RMSE: {rmse}")
print(f"R^2 Score: {r2}")
print(f"Model score is: {rf_model_score}")


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
print('The model score is:', linear_model.score(X_train, y_train))
linear_predictions = linear_model.predict(X_test)
print("Linear Regression Performance:")
print("MSE:", mean_squared_error(y_test, linear_predictions))
# print("MAE:", mean_absolute_error(y_test, linear_predictions))
print("R2 Score:", r2_score(y_test, linear_predictions))
