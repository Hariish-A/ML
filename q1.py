
# Gradient descent with one independent variable
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv(r"/content/sample_data/DIABETICS DATASET.csv")

# Normalizing/Scaling data
scaler = RobustScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

#removing outliers
Q1 = df['Outcome'].quantile(0.25)
Q3 = df['Outcome'].quantile(0.75)
IQR = Q3 - Q1
lowerbound = Q1 - 1.5 * IQR
upperbound = Q3 + 1.5 * IQR
df = df[(df['Outcome'] >= lowerbound) & (df['Outcome'] <= upperbound)]

df.corr()

# Define independent variables and the target variable
X1 = df[["Glucose"]].values  # Convert to numpy array
Y = df[["Outcome"]].values  # Convert to numpy array

X_train, X_test, Y_train, Y_test = train_test_split(X1, Y, test_size=0.3, random_state=42)

#TRAINING

# Initialize coefficients
x0 = 0
x1 = 0

# Learning rate and number of epochs
L = 0.001
epoch = 1000 # number of iterations over full dataset
m = len(Y)  # Number of samples

# Gradient Descent
for i in range(epoch):
    y_pred = x1 * X_train + x0  # Predicted values
    D0 = (-2/m) * np.sum(Y_train - y_pred)  # Derivative with respect to x0
    D1 = (-2/m) * np.sum((Y_train - y_pred) * X_train)  # Derivative with respect to x1
    x1 = x1 - L * D1  # Update x1
    x0 = x0 - L * D0  # Update x0

# Print the final coefficients
print("x1:", x1)
print("x0:", x0)

#TESTING

Y_pred = x1 * X_test + x0  # Predicted values

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("MSE:", mse)
print("R2:", r2)

# Interpreting the results:

# MSE: Lower values of MSE indicate better model performance.
# It represents the average squared difference between the predicted and actual values.

# R-squared: This value ranges from 0 to 1, with higher values indicating a better fit.
# It represents the proportion of variance in the target variable explained by the model.

# Plot the data
plt.scatter(X_train, Y_train)
plt.xlabel("Glucose")
plt.ylabel("Diabetes")
plt.show()

# Plot the regression line
plt.scatter(X_train, Y_train)
plt.plot(X_train, x1 * X_train + x0, color='red')
plt.xlabel("Glucose")
plt.ylabel("Diabetes")
plt.show()

# Gradient descent with two independent variable


# Load the dataset
df = pd.read_csv('/content/sample_data/california_housing_train.csv')

# Normalizing/Scaling data
scaler = RobustScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

df.corr()

# Define independent variables and the target variable
X1 = df[["median_income"]].values  # Convert to numpy array
X2 = df[["total_rooms"]].values  # Convert to numpy array
Y = df[["median_house_value"]].values  # Convert to numpy array

# Plot the data
plt.scatter(X1, Y)
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.show()

plt.scatter(X2, Y)
plt.xlabel("Total Rooms")
plt.ylabel("Median House Value")
plt.show()

# Initialize coefficients
x0 = 0
x1 = 0
x2 = 0

# Learning rate and number of epochs
L = 0.0001
epoch = 100000
m = len(Y)  # Number of samples

# Gradient Descent
for i in range(epoch):
    y_pred = x1 * X1 + x2 * X2 + x0  # Predicted values
    D0 = (-2/m) * np.sum(Y - y_pred)  # Derivative with respect to x0
    D1 = (-2/m) * np.sum((Y - y_pred) * X1)  # Derivative with respect to x1
    D2 = (-2/m) * np.sum((Y - y_pred) * X2)  # Derivative with respect to x2
    x1 = x1 - L * D1  # Update x1
    x2 = x2 - L * D2  # Update x2
    x0 = x0 - L * D0  # Update x0

# Print the final coefficients
print("x1:", x1)
print("x2:", x2)
print("x0:", x0)

test = pd.read_csv('/content/sample_data/california_housing_test.csv')
# Scale the data using RobustScaler
scaler = RobustScaler()
test = pd.DataFrame(scaler.fit_transform(test), columns=test.columns)

# Define independent variables and the target variable
X1_test = test[["median_income"]].values  # Convert to numpy array
X2_test = test[["total_rooms"]].values  # Convert to numpy array
Y_test = test[["median_house_value"]].values  # Convert to numpy array

Y_pred = x1 * X1_test + x2 * X2_test + x0  # Predicted values

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print("MSE:", mse)
print("R2:", r2)

# Interpreting the results:

# MSE: Lower values of MSE indicate better model performance.
# It represents the average squared difference between the predicted and actual values.

# R-squared: This value ranges from 0 to 1, with higher values indicating a better fit.
# It represents the proportion of variance in the target variable explained by the model.