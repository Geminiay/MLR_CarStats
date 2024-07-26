from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import openpyxl

def read_excel_to_matrix(file_path):
    #Load the workbook and select the active sheet
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active
    
    #Initialize an empty matrix
    data_X = []
    data_y = []

    #Iterate through the rows in the sheet
    for X in range(2, sheet.max_row + 1):
        row_data = []
        for col in range(2, sheet.max_column):
            cell_value = sheet.cell(row=X, column=col).value
            row_data.append(float(cell_value))
        data_X.append(row_data)
    
    for y in range(2, sheet.max_row + 1):
        cell_value = sheet.cell(row=y, column=7).value
        data_y.append(float(cell_value))

    array_X = np.array(data_X)
    array_y = np.array(data_y)

    #Close the workbook
    workbook.close()
    
    return array_X, array_y


def trainData(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    coefficients = model.coef_
    print("Model Coefficients:")
    for i, coef in enumerate(coefficients):
        print(f'Coefficient for feature {i+1}: {coef}')


#Define the file path
file_path = 'dataset.xlsx'

#Read the data into a matrix
X , y = read_excel_to_matrix(file_path)

print("Feature Matrix X:\n")
print(X)

print("Target Array y:")
print(y)

trainData(X, y)

