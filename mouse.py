import pandas as pd
import numpy as np
import matplotlib.pyplot as pltf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


mouse_weight = np.array([5, 7, 10, 15, 12, 20, 4, 5])
mouse_size = np.array([1, 2, 3, 5, 4, 6, 1, 1])


df = pd.DataFrame({'Weight': mouse_weight, 'Size': mouse_size})

# df = df.sort_values(by='Size', ascending=True)
print(df)

X = df[['Weight']]
y = df['Size']

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=42)


# train_shape = pd.DataFrame({'Train_X': X_train, 'Train_y': y_train})
# test_shape = pd.DataFrame({'Test_X': X_test, 'Test_y': y_test})

print(X_train)
print(y_train)


# Defind the model
model = LinearRegression()

# Feed the training data
model.fit(X_train, y_train)

# Train on test data
y_predict = model.predict(X_test)

prediction_mouse = pd.DataFrame({'Size_Test': y_test, 'Size_Predict': y_predict})
print(prediction_mouse)
# print(f"Test Size: {y_test}")
# print(f"Predict Size: {y_predict}")