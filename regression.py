from sklearn.linear_model import LinearRegression
import numpy as np
import openpyxl

dataset = openpyxl.load_workbook("database/dataset.xlsx")

for row in range(0, dataset.max_row):
    for col in dataframe1.iter_cols(1, dataset.max_column):
        X[col][row] = col[row].value

print(X)