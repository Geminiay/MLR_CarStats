from sklearn.linear_model import LinearRegression
import numpy as np
import openpyxl

def read_excel_to_matrix(file_path):
    #Load the workbook and select the active sheet
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active
    
    #Initialize an empty matrix
    matrix_X = []
    matrix_Y = []

    #Iterate through the rows in the sheet
    for X in range(2, sheet.max_row + 1):
        row_data = []
        for col in range(2, sheet.max_column):
            cell_value = sheet.cell(row=X, column=col).value
            row_data.append(cell_value)
        matrix_X.append(row_data)
    
    for Y in range(2, sheet.max_row + 1):
        row_data = []
        cell_value = sheet.cell(row=Y, column=7).value
        row_data.append(cell_value)
        matrix_Y.append(row_data)

    #Close the workbook
    workbook.close()
    
    return matrix_X, matrix_Y

#Define the file path
file_path = 'dataset.xlsx'

#Read the data into a matrix
matrix_X, matrix_Y = read_excel_to_matrix(file_path)

#Print the matrix
for X in matrix_X:
    print(X)

for Y in matrix_Y:
    print(Y)