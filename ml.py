# Import all of neccessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('kioviet.csv')
df = df.drop(['Customer_Name'], axis=1)
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
df['Year'] = df['PurchaseDate'].dt.year 
df['Month'] = df['PurchaseDate'].dt.month
df['Day'] = df['PurchaseDate'].dt.day
df = pd.get_dummies(df, columns=['DayOfWeek'])
df = df[['Year', 'Month', 'Day', 'Hour', 'DayOfWeek_Monday', 'DayOfWeek_Tuesday', 'DayOfWeek_Wednesday', 'DayOfWeek_Thursday', 'DayOfWeek_Friday', 'DayOfWeek_Saturday', 'DayOfWeek_Sunday', 'Sales']]
print(df, df.dtypes)

X = df.drop(['Sales'], axis=1)
y = df['Sales']

# Convert the pandas DataFrame and Series to NumPy arrays
X = X.to_numpy()
y = y.to_numpy()

df.to_csv('ml.csv', index=False)

# print(X,y)