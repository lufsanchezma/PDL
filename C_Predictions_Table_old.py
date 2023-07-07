import os
import torch
import numpy as np
import joblib as jl
import xlwings as xw
import sklearn as sk


# Loading required excel books and PCA data
Tiresias_Book = xw.Book('Tiresias_Book.xlsm')
pca_components, pca = jl.load(os.getcwd() + '\\Model Data' + '\\PCA')
scaler_input, scaler_output = jl.load(os.getcwd() + '\\Model Data' + '\\Scaler')
input_names, output_names = jl.load(os.getcwd() + '\\Model Data' + '\\Variables')

Components = jl.load(os.getcwd() + '\\Model Data' + '\\PCA')
Do_PCA = jl.load(os.getcwd() + '\\Model Data' + '\\PCA_2')  # Saving the transformation data
input_data_max = jl.load(os.getcwd() + '\\Model Data' + '\\input_max')  # Saving the PCA statys
output_data_max = jl.load(os.getcwd() + '\\Model Data' + '\\output_max')  # Saving the PCA statys


# Taking information for best models

output_n = int(Tiresias_Book.sheets['Predictions'].range('F4').current_region.end('down').row - 3)
input_n = int(Tiresias_Book.sheets['Predictions'].range('A4').current_region.end('down').row - 3)
Model_best = Tiresias_Book.sheets['Predictions'].range((4, 9), (4 + output_n, 9)).value
var_changes = Tiresias_Book.sheets['Sheet1'].range((4, 2), (input_n + 3, 2)).value  # Input variable "Locked" or "OK"
var_user = Tiresias_Book.sheets['Sheet1'].range((4, 6), (input_n + 3, 6)).value  # User or Tiresias_Book
var_min = Tiresias_Book.sheets['Sheet1'].range((4, 8), (input_n + 3, 8)).value  # Variable minimum value
var_max = Tiresias_Book.sheets['Sheet1'].range((4, 9), (input_n + 3, 9)).value  # Variable maximum value

lock_del = 0  # Counter to fit the "Locked" vector and the input matrix
n_predictions = int(Tiresias_Book.sheets['Predictions_Table'].range('A2').current_region.end('down').row - 1)  # Num of rows

input_test = Tiresias_Book.sheets['Predictions_Table'].range((2, 1), (n_predictions + 1, input_n)).value

# Considering the "Locked" variables are not DoF, they must be removed from the input data

input_test = input_test / np.array(input_data_max)

# Avoid PCA if Features = max features (known from B_Tiresias)
if Do_PCA:
    input_test = sk.decomposition.PCA.transform(Components, np.array(input_test))
    input_test = torch.tensor(input_test, dtype=torch.float32)
else:
    input_test = torch.tensor(input_test, dtype=torch.float32)

for n in range(output_n):

    Model_name = Model_best[n]

    if Model_name[0:7] == "Kriging":
        K_n = Model_name[-1]  # Takes the last character, which is the number of the best Kernel

        model, likelihood = torch.load(os.getcwd() + '\\Model Data' + '\\K_' + K_n + '_Var_' + str(n) + '.pt')
        model.eval()

        with torch.no_grad():
            if input_n == 1:
                input_test = torch.unsqueeze(input_test, 1)
            distribution = likelihood(model(input_test))
            prediction = distribution.mean * output_data_max[n]
            prediction = prediction.tolist()
            Tiresias_Book.sheets['Predictions_Table'].range(2, input_n + n + 1).value = prediction

    elif Model_name[0:3] == "ANN":

        ANN_n = Model_name[-1]  # Takes the last character, which is the number of the best ANN
    
        model = torch.load(os.getcwd() + '\\Model Data' + '\\ANN_' + ANN_n + '_Var_' + str(n) + '.pt')
        model.eval()

        with torch.no_grad():
            if input_n == 1:
                input_test = torch.unsqueeze(input_test, 1)
            prediction = model(input_test) * output_data_max[n]
            prediction = prediction.tolist()
            Tiresias_Book.sheets['Predictions_Table'].range(2, input_n + n + 1).value = prediction

    else:

        X_test = np.array(input_test)
        if input_n == 1:
            X_test = np.reshape(X_test, (-1, 1))
        if Model_best[n] == "Second Order Polynomial Regression":
            poly = sk.preprocessing.PolynomialFeatures(degree=2)
            X_test = poly.fit_transform(X_test)
        elif Model_best[n] == "Third Order Polynomial Regression":
            poly = sk.preprocessing.PolynomialFeatures(degree=3)
            X_test = poly.fit_transform(X_test)
        elif Model_best[n] == "Fourth Order Polynomial Regression":
            poly = sk.preprocessing.PolynomialFeatures(degree=4)
            X_test = poly.fit_transform(X_test)

        model = jl.load(os.getcwd() + '\\Model Data' + '\\' + Model_name + str(n))
        Y_test = model.predict(X_test)
        Y_test = torch.tensor(model.predict(X_test), dtype=torch.float32) * output_data_max[n]
        prediction = torch.unsqueeze(Y_test, 1)
        prediction = prediction.tolist()
        Tiresias_Book.sheets['Predictions_Table'].range(2, input_n + n + 1).value = prediction
