import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import linear_model
# Load dataset, no header
filename = 'ML_2200_MultipleLinearRegression.csv'
raw_data = open(filename, 'rt')
data = np.loadtxt(raw_data, delimiter=",")

# x y
matrix_X = data[:,2:4]
matrix_Y= data[:,1]
# Create MuLTIPLE linear regression object
modelM = linear_model.LinearRegression()
modelM.fit(matrix_X,matrix_Y)
# The coefficients
print('Coefficients:')
print('B0: \n', modelM.intercept_)
print('B1: \n', modelM.coef_[0])
print('B2: \n', modelM.coef_[1])
# R^2
print('R^2: \n', modelM.score(matrix_X, matrix_Y))