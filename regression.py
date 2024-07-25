from sklearn.linear_model import LinearRegression
import numpy as np
import openpyxl

def read_excel_to_matrix(file_path):
    #Load the workbook and select the active sheet
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active
    
    #Initialize an empty matrix
    data_matrix = []

    #Iterate through the rows in the sheet
    for X in range(2, sheet.max_row + 1):  # Assuming 1-based index
        row_data = []
        for col in range(2, sheet.max_column + 1):  # Assuming 1-based index
            cell_value = sheet.cell(row=X, column=col).value
            row_data.append(cell_value)
        data_matrix.append(row_data)
    
    #Close the workbook
    workbook.close()
    
    return data_matrix

#Define the file path
file_path = 'dataset.xlsx'

#Read the data into a matrix
matrix = read_excel_to_matrix(file_path)

#Print the matrix
for X in matrix:
    print(X)