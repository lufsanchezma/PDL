import os
import torch
import numpy as np
import joblib as jl
import xlwings as xw
import sklearn as sk


# Loading required excel books and PCA data
Tiresias = xw.Book('Tiresias.xlsm')
Components = jl.load(os.getcwd() + '\\Model Data' + '\\PCA')
Do_PCA = jl.load(os.getcwd() + '\\Model Data' + '\\PCA_2')  # Saving the transformation data
input_data_max = jl.load(os.getcwd() + '\\Model Data' + '\\input_max')  # Saving the PCA statys
output_data_max = jl.load(os.getcwd() + '\\Model Data' + '\\output_max')  # Saving the PCA statys


# Taking information for best models

output_n = int(Tiresias.sheets['Predictions'].range('F4').current_region.end('down').row - 3)
input_n = int(Tiresias.sheets['Predictions'].range('A4').current_region.end('down').row - 3)
Model_best = Tiresias.sheets['Predictions'].range((4, 9), (4 + output_n, 9)).value

lock_del = 0  # Counter to fit the "Locked" vector and the input matrix
input_test = torch.tensor(Tiresias.sheets['Predictions'].range((4, 4), (4 + input_n - 1, 4)).value)

# Considering the "Locked" variables are not DoF, they must be removed from the input data

input_test = input_test / np.array(input_data_max)

# Avoid PCA if Features = max features (known from B_Tiresias)
if Do_PCA:
    input_test = sk.decomposition.PCA.transform(Components, [np.array(input_test)])
    input_test = torch.tensor(input_test, dtype=torch.float32)

for n in range(output_n):

    Model_name = Model_best[n]

    if Model_name[0:7] == "Kriging":
        K_n = Model_name[-1]  # Takes the last character, which is the number of the best Kernel

        model, likelihood = torch.load(os.getcwd() + '\\Model Data' + '\\K_' + K_n + '_Var_' + str(n) + '.pt')
        model.eval()

        with torch.no_grad():

            distribution = likelihood(model(torch.unsqueeze(input_test, 0)))
            prediction = distribution.mean * output_data_max[n]
            Tiresias.sheets['Predictions'].range(4 + n, 10).value = prediction

    elif Model_name[0:3] == "ANN":

        ANN_n = Model_name[-1]  # Takes the last character, which is the number of the best ANN
    
        model = torch.load(os.getcwd() + '\\Model Data' + '\\ANN_' + ANN_n + '_Var_' + str(n) + '.pt')
        model.eval()

        with torch.no_grad():
            prediction = model(input_test) * output_data_max[n]
            Tiresias.sheets['Predictions'].range(4 + n, 10).value = prediction

    else:

        X_test = np.array(input_test)

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

        if Do_PCA:
            Y_test = np.dot(model.predict(X_test), np.array(output_data_max[n]))
        else:
            Y_test = np.dot(model.predict([X_test]), np.array(output_data_max[n]))

        Tiresias.sheets['Predictions'].range(4 + n, 10).value = np.round(Y_test, decimals=3)
