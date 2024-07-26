from sklearn.linear_model import LinearRegression
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
            row_data.append(cell_value)
        data_X.append(row_data)
    
    for y in range(2, sheet.max_row + 1):
        cell_value = sheet.cell(row=y, column=7).value
        data_y.append(cell_value)

    array_X = np.array(data_X)
    array_y = np.array(data_y)

    #Close the workbook
    workbook.close()
    
    return array_X, array_y

#Define the file path
file_path = 'dataset.xlsx'

#Read the data into a matrix
X , y = read_excel_to_matrix(file_path)


print("Feature Matrix X:\n")
print(X)

print("Target Array y:")
print(y)