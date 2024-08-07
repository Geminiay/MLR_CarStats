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

# Updated Constraints
def constraint1(beta):
    return np.sin(np.radians(beta[4])) * beta[1] - 575.75

def constraint2(beta):
    return np.sin(np.radians(beta[3])) * beta[0] - 690.33

def constraint3(beta):
    return np.cos(np.radians(beta[3])) * beta[0] + np.cos(np.radians(beta[4])) * beta[1] + beta[2] - 3001.2

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

# Check if initial guess satisfies constraints
print("Initial guess feasibility:")
print("Constraint 1:", constraint1(initial_guess))
print("Constraint 2:", constraint2(initial_guess))
print("Constraint 3:", constraint3(initial_guess))

# Define the constraints in the form required by 'minimize'
constraints = [
    {'type': 'eq', 'fun': constraint1},
    {'type': 'eq', 'fun': constraint2},
    {'type': 'eq', 'fun': constraint3}
]

# Optimize
result = minimize(objective, initial_guess, constraints=constraints, method='SLSQP', options={'disp': True})

# Display results
if result.success:
    optimized_parameters = result.x
    print("Optimized parameters:", optimized_parameters)
else:
    print("Optimization failed:", result.message)
