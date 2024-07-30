import numpy as np
from scipy.optimize import minimize
import pandas as pd
from sklearn.preprocessing import StandardScaler

def read_excel_to_matrix(file_path):
    # Load the data using pandas
    data = pd.read_excel(file_path)
    
    # Assuming the 2nd to 6th columns are the features and the 7th column is the target
    data_X = data.iloc[:, 1:6].values  # Features
    data_y = data.iloc[:, 6].values    # Target

    return np.array(data_X), np.array(data_y)

# Objective function: Mean Squared Error
def objective(beta):
    predictions = beta[0] + np.dot(X, beta[1:])
    mse = np.mean((y - predictions) ** 2)
    return mse

# Combined Constraints
def aggregated_constraint1(beta):
    return np.sum(np.sin(np.radians(X[:, 3])) * X[:, 2] - np.sin(np.radians(X[:, 1])) * X[:, 0] - 20.73)

def aggregated_constraint2(beta):
    return np.sum(np.cos(np.radians(X[:, 1])) * X[:, 0] + np.cos(np.radians(X[:, 3])) * X[:, 2] + X[:, 4] - 3001.20)

# Define the file path
file_path = 'dataset.xlsx'

# Read the data into a matrix
X, y = read_excel_to_matrix(file_path)

# Print the matrices before scaling
print("Feature matrix (X) before scaling:\n", X)
print("Target vector (y):\n", y)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Print the matrices after scaling
print("Feature matrix (X) after scaling:\n", X)

# Initial guess for the parameters
initial_guess = np.ones(6) * 0.1  # Small non-zero initial guess

# Define the constraints in the form required by 'minimize'
constraints = [
    {'type': 'eq', 'fun': aggregated_constraint1},
    {'type': 'eq', 'fun': aggregated_constraint2}
]

# Optimize
result = minimize(objective, initial_guess, constraints=constraints)

# Display results
if result.success:
    optimized_parameters = result.x
    print("Optimized parameters:", optimized_parameters)
else:
    print("Optimization failed:", result.message)
