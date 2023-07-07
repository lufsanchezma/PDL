import os
import sys
import time
import torch
import warnings
import gpytorch
import numpy as np
import joblib as jl
import xlwings as xw
import sklearn as sk
import torch.nn as nn
import sklearn.compose as skc
import sklearn.ensemble as ske
import sklearn.model_selection as skm
import matplotlib.pyplot as plt

from math import ceil
from tabulate import tabulate
from F_Functions import Net1, Net2, Net3, MultitaskGPModel  # Import the ANN and Kriging definitions

warnings.filterwarnings("ignore")

# DATA COLLECTION FROM EXCEL DATASHEETS


# Calling the Tiresias workbooks
start_time = time.time()
Tiresias = xw.Book('Tiresias.xlsm')
Tiresias_Data = xw.Book('Tiresias_Data.xlsx')
Tiresias_Models = xw.Book('Tiresias_models.xlsx')

# Recovering the number of variables and simulations from Tiresias workbook
data_source = int(Tiresias.sheets['Sheet1'].range(2, 12).value)  # Hysys or Real Plant Data
Features = int(Tiresias.sheets['Sheet1'].range('M25').value)
input_n = int(Tiresias.sheets['Sheet1'].range('C4').current_region.end('down').row - 3)
n_components = int(Tiresias.sheets['Sheet1'].range('M17').value)
n_folds = int(Tiresias.sheets['Sheet1'].range('M24').value)  # Number of sets to divide the data in for C.V.
output_n = int(Tiresias.sheets['Sheet1'].range('Q4').current_region.end('down').row - 3)

var_changes = Tiresias.sheets['Sheet1'].range((4, 2), (input_n + 3, 2)).value  # Input variable "Locked" or "OK"
var_user = Tiresias.sheets['Sheet1'].range((4, 6), (input_n + 3, 6)).value  # User or Tiresias
var_min = Tiresias.sheets['Sheet1'].range((4, 8), (input_n + 3, 8)).value  # Variable minimum value
var_max = Tiresias.sheets['Sheet1'].range((4, 9), (input_n + 3, 9)).value  # Variable maximum value

regression_yn = Tiresias.sheets['RUN'].range('E15').value  # Train regression models y/n
ANN_yn = Tiresias.sheets['RUN'].range('E16').value  # Train ANNs and number of ANNs to train
K_yn = Tiresias.sheets['RUN'].range('E17').value  # Train Kriging model y/n

# Fetching data from Tiresias_Data
# Fetching data from Tiresias_Data
n_simulations = int(Tiresias_Data.sheets['Sheet1'].range('A2').current_region.end('down').row - 1)  # Num of data rows

data = Tiresias_Data.sheets['Sheet1'].range((2, input_n + 1), (n_simulations + 1, input_n + output_n)).value  # Outputs
data_2 = Tiresias_Data.sheets['Sheet1'].range((2, 1), (n_simulations + 1, input_n + output_n)).value  # Complete dataset

# If only one exit, data must be fixed so its read as matrix instead of vector
if output_n == 1:
    data = np.array([data])
    data = data.T

# Defining the output variables names from Sheet1 in Tiresias datasheet
name_output = [0] * output_n

for n in range(output_n):
    name_output[n] = Tiresias.sheets['Sheet1'].range(n + 4, 17).value + '.' + Tiresias.sheets['Sheet1'].range(n + 4,
                                                                                                              18).value

# DATA CLEANING

# Remove non-convergent simulations (String or Negative P/F/Compositions)
RowRemoved = 0
j = 0

print("Data Cleaning in process...")

if data_source == 2:  # If real plant data

    while j < n_simulations:
        sys.stdout.write("\rValidating data... (%i/%i)\033[K" % (j + 1, n_simulations))
        sys.stdout.flush()
        for i in range(input_n + output_n):  # In each row checks data availability
            if data_2[j][i] is None:  # If empty data, deletes the rest of the information
                Tiresias_Data.sheets['Sheet1'].range(str(j + 2) + ':' + str(n_simulations + 1)).api.Delete()
                RowRemoved = n_simulations - j
                n_simulations = int(Tiresias_Data.sheets['Sheet1'].range('A2').current_region.end('down').row - 1)
                data_2 = Tiresias_Data.sheets['Sheet1'].range((2, 1), (n_simulations + 1, input_n + output_n)).value
                j -= RowRemoved
                j -= 1
        j += 1

else:

    while j < n_simulations:
        sys.stdout.write("\rValidating data... (%i/%i)\033[K" % (j + 1, n_simulations))
        sys.stdout.flush()
        for i in range(output_n):  # In each row checks convergence and data-type issues
            if data[j][i] == -32767:  # Exact value observed when Hysys returns non-convergence
                Tiresias_Data.sheets['Sheet1'].range(str(j + 2) + ':' + str(j + 2)).api.Delete()
                RowRemoved += 1
                j -= 1
                n_simulations = int(
                    Tiresias_Data.sheets['Sheet1'].range('A2').current_region.end('down').row - 1)
                data = Tiresias_Data.sheets['Sheet1'].range((2, input_n + 1),
                                                            (n_simulations + 1, input_n + output_n)).value
                # If only one exit, data must be fixed so its read as matrix instead of vector
                if output_n == 1:
                    data = np.array([data])
                    data = data.T

            elif data[j][i] < 0:  # To check that data is physically meaningful. Only T and heat can be negative.
                name_dummy = name_output[i]
                if name_dummy[-1] != "T" and name_dummy[-1] != "Q":
                    Tiresias_Data.sheets['Sheet1'].range(str(j + 2) + ':' + str(j + 2)).api.Delete()
                    RowRemoved += 1
                    j -= 1
                    n_simulations = int(
                        Tiresias_Data.sheets['Sheet1'].range('A2').current_region.end('down').row - 1)
                    data = Tiresias_Data.sheets['Sheet1'].range((2, input_n + 1),
                                                                (n_simulations + 1, input_n + output_n)).value
                    # If only one exit, data must be fixed so its read as matrix instead of vector
                    if output_n == 1:
                        data = np.array([data])
                        data = data.T
        j += 1

print("\nData has ben validated. %i rows were removed." % RowRemoved)
Tiresias_Data.save()

# Using the known amount of inputs and outputs to recall variables and their names from Tiresias_Data
input_varname = Tiresias_Data.sheets['Sheet1'].range((1, 1), (1, input_n)).value
output_varname = Tiresias_Data.sheets['Sheet1'].range((1, input_n + 1), (1, input_n + output_n)).value

input_data_tot = np.array(Tiresias_Data.sheets['Sheet1'].range((2, 1), (n_simulations + 1, input_n)).value)
output_data_tot = np.array(Tiresias_Data.sheets['Sheet1'].range((2, input_n + 1),
                                                                (n_simulations + 1, input_n + output_n)).value)

np.random.shuffle(input_data_tot)  # Uses NumPy to shuffle data before using it, avoiding biases
np.random.shuffle(output_data_tot)  # Uses NumPy to shuffle data before using it, avoiding biases

input_data_tot = torch.tensor(input_data_tot, dtype=torch.float32)
output_data_tot = torch.tensor(output_data_tot, dtype=torch.float32)

# REMOVING NON-CHANGING INPUT VARIABLES

# Considering the "Locked" variables are not DoF, they must be removed from the input data
# Variables with maximum and minimum values equal are also deleted
lock_del = 0  # Counter to fit the "Locked" vector and the input matrix

for n in range(len(input_data_tot[0])):
    if input_n == 1:
        if var_changes == 2 or (var_user == 2 and var_min == var_max):
            print('There is only one input variable and is not changing. Please check your input data.')
            quit()
    else:
        if var_changes[n] == 2 or (var_user[n] == 2 and var_min[n] == var_max[n]):
            input_data_tot = np.delete(input_data_tot, n - lock_del, 1)
            input_varname = np.delete(input_varname, n - lock_del, 0)
            lock_del += 1

print("\n%i variables found to be non-changing in Tiresias.xlsm. Their values are not considered as inputs." % lock_del)
print("Effective number of samples: %i" % n_simulations)
print("Number of input variables: %i" % (input_n - lock_del))

# If only one variable, tensors must be set to column vectors to avoid errors


if output_n == 1:
    output_data_tot = torch.unsqueeze(output_data_tot, 1)

if input_n == 1:
    input_data_tot = torch.unsqueeze(input_data_tot, 1)

# DATA NORMALIZATION

if input_n == 1:
    Col_Data = 1
else:
    Row_Data, Col_Data = input_data_tot.shape

input_data_max = torch.zeros(Col_Data, dtype=torch.float32)

for n in range(Col_Data):

    if abs(torch.max(input_data_tot[:, n])) >= abs(torch.min(input_data_tot[:, n])):
        input_data_max[n] = torch.max(input_data_tot[:, n])
    else:
        input_data_max[n] = torch.min(input_data_tot[:, n])

    if input_data_max[n] == 0:
        input_data_max[n] = 1E-15

if output_n == 1:
    Col_Data = 1
else:
    Row_Data, Col_Data = output_data_tot.shape

output_data_max = torch.zeros(Col_Data, dtype=torch.float32)

for n in range(Col_Data):

    if abs(torch.max(output_data_tot[:, n])) >= abs(torch.min(output_data_tot[:, n])):
        output_data_max[n] = torch.max(output_data_tot[:, n])
    else:
        output_data_max[n] = torch.min(output_data_tot[:, n])

    if output_data_max[n] == 0:
        output_data_max[n] = 1E-15

input_data_norm = np.divide(input_data_tot, input_data_max)
Tiresias_Data.sheets['Normalized'].range(1, 2).value = np.array(input_data_norm)
output_data_norm = np.divide(output_data_tot, output_data_max)

# FEATURE EXTRACTION

# Maximum number of features cannot exceed the number of simulations or input variables
if Features > min((input_n - lock_del), n_simulations):
    print("Number of features chosen by the user: %i" % Features)
    Features = min((input_n - lock_del), n_simulations)
    print("The number of Features has been set to %i, which is the maximum possible number of features for this case \n"
          % Features)
else:
    print("Number of features: %i\n" % Features)

Components = sk.decomposition.PCA(n_components=Features).fit(input_data_norm)
PCA_weight = Components.components_
mean_weight = [0] * len(input_data_norm[0])
names = [0] * len(input_data_norm[0])

for n in range(len(input_data_norm[0])):
    mean_weight[n] = np.mean(PCA_weight[:, n])

mean_weight = np.array(mean_weight)

order = np.argsort(mean_weight)
dummy_ = mean_weight
mean_weight = np.sort(mean_weight)

# PUTTING VARIABLE NAMES IN PREDICTIONS SHEET

for n in range(len(input_varname)):
    input_varname_temp = input_varname[n].split('.')

    Tiresias.sheets['Predictions_Table'].range(1, n + 1).value = input_varname[n]

    Tiresias.sheets['Predictions'].range(n + 4, 1).value = input_varname_temp[0]
    Tiresias.sheets['Predictions'].range(n + 4, 2).value = input_varname_temp[1]
    Tiresias.sheets['Predictions'].range(n + 4, 4).value = 0

    if input_varname_temp[1] == 'T':
        Tiresias.sheets['Predictions'].range(n + 4, 3).value = "C"
    elif input_varname_temp[1] == 'P':
        Tiresias.sheets['Predictions'].range(n + 4, 3).value = "bar"
    elif input_varname_temp[1] == 'F':
        Tiresias.sheets['Predictions'].range(n + 4, 3).value = "kg/h"
    else:
        Tiresias.sheets['Predictions'].range(n + 4, 3).value = "Mol Frac"

for n in range(output_n):
    Tiresias.sheets['Predictions'].range(n + 4, 6).value = Tiresias.sheets['Sheet1'].range(n + 4, 17).value
    Tiresias.sheets['Predictions'].range(n + 4, 7).value = Tiresias.sheets['Sheet1'].range(n + 4, 18).value
    Tiresias.sheets['Predictions'].range(n + 4, 8).value = Tiresias.sheets['Sheet1'].range(n + 4, 19).value

    Tiresias.sheets['Predictions_Table'].range(1, n + len(input_varname) + 1).value = output_varname[n]

if input_n == 1:
    names = input_varname
else:
    for n in range(len(PCA_weight[0])):
        names[n] = input_varname[order[n]]

font = {'size': 10}
plt.rc('font', **font)
plt.rcParams['figure.constrained_layout.use'] = True

plt.bar(names, mean_weight)
plt.xticks(rotation=90)

plt.ylabel('Average PCA weight')

plt.axhline(y=0.02, color='r', linestyle="--", linewidth=0.7)
plt.axhline(y=-0.02, color='r', linestyle="--", linewidth=0.7)

# CHECK THE DECOMPOSITION, SOMETHING WEIRD

# If features = max features avoid PCA
if Features == min((input_n - lock_del), n_simulations):
    Do_PCA = False
else:
    Do_PCA = True
    input_data_norm = sk.decomposition.PCA.transform(Components, input_data_norm)

jl.dump(Components, os.getcwd() + '\\Model Data' + '\\PCA')  # Saving the transformation data
jl.dump(Do_PCA, os.getcwd() + '\\Model Data' + '\\PCA_2')  # Saving the PCA status
jl.dump(input_data_max, os.getcwd() + '\\Model Data' + '\\input_max')  # Saving the input maximums
jl.dump(output_data_max, os.getcwd() + '\\Model Data' + '\\output_max')  # Saving the output maximums

# Splitting (before), now just array redefinition
# xtrain, xtest, ytrain, ytest = skm.train_test_split(input_data_tot, output_data_tot, test_size=0.0, random_state=4)

dummy = input_data_norm
ytrain = output_data_norm
n_metrics_columns = 10  # RMSE, STD, MAE, STD, R2 Train and Test
n_reg_models = 9  # Regression models to test
total_reg_metrics = n_metrics_columns * n_reg_models  # Sheet -Results- in Tiresias_Models total number of columns

if regression_yn == 1:

    # SKLEARN - REGRESSION MODELS

    Time_Dummy_reg = time.time()

    Scores_reg = np.zeros([n_folds, n_metrics_columns * output_n], dtype=float)  # Finds the overall metrics
    Results_reg = np.zeros([output_n, total_reg_metrics], dtype=float)  # Matrix displayed in -Results- with all metrics
    Reg_tree = 0.0  # To quickly test Trees or SVR Regularization (ccp_alpha or C). Default 0
    Reg_alpha = Reg_tree  # Same, for regularization of polynomials. Default 0
    print('Regression models fitting and evaluation in progress...')

    for n in range(output_n):

        for Selected_Model in range(n_reg_models):

            model_name = "No model"  # In case of errors
            model_0 = "No model"
            poly = sk.preprocessing.PolynomialFeatures(degree=1)
            xtrain = dummy  # To reset the variables, as polynomial features modifies them

            # Regression model selection
            if Selected_Model == 0:
                model_0 = ske.RandomForestRegressor(n_estimators=100, ccp_alpha=Reg_tree)  # Default 100 , 0
                model_name = "Random Forest"
            if Selected_Model == 1:
                model_0 = sk.linear_model.Ridge(alpha=Reg_tree)  # Default 0}
                model_name = "Linear Regression"
            if Selected_Model == 2:
                model_0 = sk.svm.SVR(C=1.0 - Reg_tree)  # Default 1
                model_name = "Support Vector Regression (SVR)"
            if Selected_Model == 3:
                model_0 = ske.GradientBoostingRegressor(max_depth=3, ccp_alpha=Reg_tree)  # Default 3, 0
                model_name = "Gradient Boost Regression"
            if Selected_Model == 4:
                model_0 = ske.AdaBoostRegressor(sk.tree.DecisionTreeRegressor(max_depth=None,
                                                                              ccp_alpha=Reg_tree))  # Default None, 0
                model_name = "Ada Boost Regression"
            if Selected_Model == 5:
                model_0 = sk.tree.DecisionTreeRegressor(max_depth=None, ccp_alpha=Reg_tree)  # Default None , 0
                model_name = "Decision Tree Regression"
            if Selected_Model == 6:
                poly = sk.preprocessing.PolynomialFeatures(
                    degree=2)  # Second order polynomial needs matrix transformation
                xtrain = poly.fit_transform(xtrain)
                model_0 = sk.linear_model.Ridge(alpha=Reg_alpha)  # Default 0
                model_name = "Second Order Polynomial Regression"
            if Selected_Model == 7:
                poly = sk.preprocessing.PolynomialFeatures(degree=3)
                xtrain = poly.fit_transform(xtrain)
                model_0 = sk.linear_model.Ridge(alpha=Reg_alpha)  # Default 0
                model_name = "Third Order Polynomial Regression"
            if Selected_Model == 8:
                poly = sk.preprocessing.PolynomialFeatures(degree=4)
                xtrain = poly.fit_transform(xtrain)
                model_0 = sk.linear_model.Ridge(alpha=Reg_alpha)  # Default 0
                model_name = "Fourth Order Polynomial Regression"

            sys.stdout.write(
                "\r\rVariable %i of %i (%s). Actual model: %s\033[K" % (n + 1, output_n, name_output[n], model_name))
            sys.stdout.flush()

            # Data normalization and Cross Validation with n_folds Folds
            model = model_0
            cv_skf = skm.KFold(n_splits=n_folds, shuffle=True, random_state=4)

            model_scores = skm.cross_validate(model, xtrain, ytrain[:, n], cv=cv_skf, scoring=[
                'neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
                                              return_train_score=True)

            sorted(model_scores.keys())
            model.fit(xtrain, ytrain[:, n])

            jl.dump(model,
                    os.getcwd() + '\\Model Data' + '\\' + model_name + str(n))  # Saving model parameters in folder

            # Fills the metrics for the cross validation of the regression models in the Results matrix
            Results_reg[n, Selected_Model * n_metrics_columns + 0] = np.mean(
                abs(model_scores['train_neg_mean_squared_error'])) ** (1 / 2)
            Results_reg[n, Selected_Model * n_metrics_columns + 1] = np.std(
                abs(model_scores['train_neg_mean_squared_error'])) ** (1 / 2)
            Results_reg[n, Selected_Model * n_metrics_columns + 2] = np.mean(
                abs(model_scores['train_neg_mean_absolute_error']))
            Results_reg[n, Selected_Model * n_metrics_columns + 3] = np.std(
                abs(model_scores['train_neg_mean_absolute_error']))
            Results_reg[n, Selected_Model * n_metrics_columns + 4] = np.mean(abs(model_scores['train_r2']))
            Results_reg[n, Selected_Model * n_metrics_columns + 5] = np.mean(
                abs(model_scores['test_neg_mean_squared_error'])) ** (1 / 2)
            Results_reg[n, Selected_Model * n_metrics_columns + 6] = np.std(
                abs(model_scores['test_neg_mean_squared_error'])) ** (1 / 2)
            Results_reg[n, Selected_Model * n_metrics_columns + 7] = np.mean(
                abs(model_scores['test_neg_mean_absolute_error']))
            Results_reg[n, Selected_Model * n_metrics_columns + 8] = np.std(
                abs(model_scores['test_neg_mean_absolute_error']))
            Results_reg[n, Selected_Model * n_metrics_columns + 9] = np.mean(abs(model_scores['test_r2']))

    sys.stdout.write("\r\rFitting for all the regression models has finished!\033[K\n\n")
    sys.stdout.flush()

    Times_reg = time.time() - Time_Dummy_reg

    # Sending results to Tiresias_Models datasheet
    Tiresias_Models.sheets['Regression'].range(6, 1).value = np.reshape(name_output, (output_n, -1))
    Tiresias_Models.sheets['Regression'].range(1, 2).value = n_folds
    Tiresias_Models.sheets['Regression'].range(6, 2).value = Results_reg

    for n in range(len(output_data_max)):
        Tiresias_Models.sheets['Regression'].range(6 + n, 98).value = output_data_max[n]

else:

    Results_reg = np.full([output_n, total_reg_metrics], 1000, dtype=float)  # Define a high error as not used.
    Tiresias_Models.sheets['Regression'].range(6, 2).value = Results_reg

# ANN's DEFINITION

if ANN_yn == 1:  # If ANNs will not be trained

    Results_ANN = np.full([output_n, 10 * n_metrics_columns], 1000, dtype=float)
    Tiresias_Models.sheets['ANN'].range(6, 2).value = Results_ANN

else:

    ANN_Tot = int(ANN_yn - 1)  # Number of ANN's to test selected by user
    Times_ANN_names = np.zeros(ANN_Tot, dtype=object)  # Array containing the execution time for each ANN
    Times_ANN_times = np.zeros(ANN_Tot, dtype=float)  # Array containing the execution time for each ANN
    Scores_ANN = np.zeros([n_folds, 6 * output_n], dtype=float)  # Finds the overall metrics

    total_ANN_metrics = n_metrics_columns * ANN_Tot  # Sheet -Results- in Tiresias_Models total number of columns
    Results_ANN = np.zeros([output_n, total_ANN_metrics], dtype=float)  # Matrix displayed in -Results- with all metrics

    print("The Artificial Neural Networks (ANN's) are being trained...")
    rows, cols = input_data_norm.shape
    rows = rows - rows % n_folds
    fold_size = int(rows / n_folds)

    # Defining the ANN's to evaluate

    Time_Dummy_ANN = time.time()
    count_diverge_ANN = 0
    count_converge_ANN = 0
    count_max_ANN = 0

    input_data_norm = torch.tensor(input_data_norm, dtype=torch.float32)
    output_data_norm = torch.tensor(output_data_norm, dtype=torch.float32)

    # TRAINING OF THE ANN's

    for ANN_n in range(ANN_Tot):

        for cont in range(n_folds + 1):

            # Manual division of the data set (for Cross Validation)

            if cont != n_folds:
                # Data is divided in n k-folds. Each fold used once as test set and n-1 as training set.
                # input_data and output_data are the n-1 k-folds used to train the C.V. fold.
                # input_test and output_test are the data of the fold used to test the C.V. training
                input_test_norm = input_data_norm[cont * fold_size:(cont + 1) * fold_size, :]
                output_test_norm = output_data_norm[cont * fold_size:(cont + 1) * fold_size, :]
                input_dummy_0 = input_data_norm[0:cont * fold_size, :]
                input_dummy_1 = input_data_norm[(cont + 1) * fold_size:rows, :]
                output_dummy_0 = output_data_norm[0:cont * fold_size, :]
                output_dummy_1 = output_data_norm[(cont + 1) * fold_size:rows, :]
                input_data = torch.cat((input_dummy_0, input_dummy_1))
                output_data = torch.cat((output_dummy_0, output_dummy_1))

            else:
                # Once C.V. done, use the whole dataset to train the ANN (improve future predictions)
                input_data = input_data_norm
                output_data = output_data_norm

            # TRAINING OF THE ACTUAL ANN (Iterative procedure)

            for n in range(output_n):

                # The model is selected depending on the actual value of the ANN_n counter

                if ANN_n == 0:

                    # ANN 0: Basic linear model ANN (no hidden layers)

                    model = nn.Linear(Features, 1)

                elif ANN_n == 1:

                    # ANN 1: ANN with one hidden layer with same input_size elements

                    model = Net1(input_size=Features, hidden_size=Features, num_classes=1)

                elif ANN_n == 2:

                    # ANN 2: ANN with one hidden layer with twice input_size elements

                    model = Net1(input_size=Features, hidden_size=Features * 2, num_classes=1)

                elif ANN_n == 3:

                    # ANN 3: ANN with one hidden layer with half input_size elements

                    model = Net1(input_size=Features, hidden_size=ceil(Features / 2), num_classes=1)

                elif ANN_n == 4:

                    # ANN 4: ANN with two hidden layers with same input_size elements

                    model = Net2(input_size=Features, hidden_size1=Features, hidden_size2=Features, num_classes=1)

                elif ANN_n == 5:

                    # ANN 5: ANN with two hidden layers with decreasing elements

                    model = Net2(input_size=Features, hidden_size1=ceil(Features / 2), hidden_size2=ceil(Features / 3),
                                 num_classes=1)

                elif ANN_n == 6:

                    # ANN 6: ANN with two hidden layers with increasing then decreasing number of elements

                    model = Net2(input_size=Features, hidden_size1=Features * 2, hidden_size2=Features,
                                 num_classes=1)

                elif ANN_n == 7:

                    # ANN 7: ANN with two hidden layers with increasing number of elements

                    model = Net2(input_size=Features, hidden_size1=Features * 2, hidden_size2=Features * 3,
                                 num_classes=1)

                elif ANN_n == 8:

                    # ANN 8: ANN with two hidden layers with decreasing then increasing number of elements

                    model = Net2(input_size=Features, hidden_size1=ceil(Features / 2), hidden_size2=Features,
                                 num_classes=1)

                elif ANN_n == 9:

                    # ANN 9: ANN with three hidden layers n_input - 3 - n_input - 3 - n_ output (Encoder - Decoder)

                    model = Net3(input_size=Features, hidden_size1=3, hidden_size2=Features, hidden_size3=3,
                                 num_classes=1)

                # Restarting the ANN convergence parameters for each output variable in each ANN
                learning_rate = 0.1
                model_max = 100000  # Maximum number of iterations for the ANN training
                model_threshold = 1e-4  # Minimum difference of loss between iterations (to consider "not changing")
                patience = 10  # Maximum number of iterations with divergence, before stopping iterations
                last_loss = 1000  # Initializing the lost of the previous iteration, to check if model is diverging
                trigger_times = 0  # Number of iterations with divergence (counter start)
                trigger_no_change = 0  # Number of iterations with change smaller than threshold (Counter start)
                model_count = 1  # Counter for the number of iterations of the ANN
                _ = 0  # Dummy variable. 0 = Keep iterating, 1 = Stop iterations.

                # Initializing parameters for iteration of the models
                model.train()
                loss = nn.MSELoss()  # Mean Squared Error (MSEloss) or MAE (L1Loss), calculated directly by PyTorch
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Stochastic gradient descent

                # Training of the ANN
                while _ < 1:
                    optimizer.zero_grad()  # Cleans gradient so no accumulation occurs
                    # Prediction (Forward step)
                    output_pred = model(input_data)
                    # Loss (Computes the difference with the real value)
                    l = loss(torch.unsqueeze(output_data[:, n], 1), output_pred)  # Un-Squeeze to match dimensions
                    # Computation of the gradients by backpropagation
                    l.backward()
                    # Optimization by choosing a new point by SGD, depending on the computed gradients
                    optimizer.step()
                    model_count += 1
                    if model_count % 1 == 0:  # Change 1 to X to print every X iterations
                        if cont != n_folds:
                            sys.stdout.write(
                                "\rANN %i of %i Cross validation in progress (%i/%i)... Variable %s... Actual error = %f (Iteration %i)\033[K"
                                % (ANN_n + 1, ANN_Tot, cont + 1, n_folds, name_output[n], l, model_count))
                            sys.stdout.flush()
                        else:
                            sys.stdout.write(
                                "\rANN %i of %i parameters evaluation in progress... Variable %s... Actual error = %f (Iteration %i)\033[K" %
                                (ANN_n + 1, ANN_Tot, name_output[n], l, model_count))
                            sys.stdout.flush()

                    if model_count >= model_max:
                        _ = 1
                        count_max_ANN += 1

                    # Early stopping 1 : If in total 10 iterations diverging, stop training
                    if l > last_loss:
                        trigger_times += 1

                    # Early stopping 2 : If 3 iterations not changing (w.r.t. threshold), stop iterations
                    if (abs(l - last_loss)) <= model_threshold:
                        trigger_no_change += (patience / 3) + 1
                    else:
                        trigger_no_change = 0

                    if trigger_times >= patience:
                        _ = 1
                        count_diverge_ANN += 1
                    if trigger_no_change >= patience:
                        _ = 1
                        count_converge_ANN += 1

                    last_loss = l

                model.eval()  # Sets the model in evaluation mode

                with torch.no_grad():

                    if cont != n_folds:
                        ypred_model = np.array(model(input_test_norm))  # Reversing normalization
                        ypred_model = np.ndarray.flatten(ypred_model)
                        # ypred_model = np.array(model(input_test_norm))
                        ytrain_model = np.array(model(input_data))  # Reversing normalization
                        ytrain_model = np.ndarray.flatten(ytrain_model)
                        # ytrain_model = np.array(model(input_data))

                    output_test_norm_np = np.array(output_test_norm)
                    output_data_np = np.array(output_data)

                if cont == n_folds:
                    # When Cross Validation is done, the metrics are obtained

                    Results_ANN[n, ANN_n * n_metrics_columns + 0] = np.mean(Scores_ANN[:, n * 6 + 0])
                    Results_ANN[n, ANN_n * n_metrics_columns + 1] = np.std(Scores_ANN[:, n * 6 + 0])
                    Results_ANN[n, ANN_n * n_metrics_columns + 2] = np.mean(Scores_ANN[:, n * 6 + 1])
                    Results_ANN[n, ANN_n * n_metrics_columns + 3] = np.std(Scores_ANN[:, n * 6 + 1])
                    Results_ANN[n, ANN_n * n_metrics_columns + 4] = np.mean(Scores_ANN[:, n * 6 + 2])
                    Results_ANN[n, ANN_n * n_metrics_columns + 5] = np.mean(Scores_ANN[:, n * 6 + 3])
                    Results_ANN[n, ANN_n * n_metrics_columns + 6] = np.std(Scores_ANN[:, n * 6 + 3])
                    Results_ANN[n, ANN_n * n_metrics_columns + 7] = np.mean(Scores_ANN[:, n * 6 + 4])
                    Results_ANN[n, ANN_n * n_metrics_columns + 8] = np.std(Scores_ANN[:, n * 6 + 4])
                    Results_ANN[n, ANN_n * n_metrics_columns + 9] = np.mean(Scores_ANN[:, n * 6 + 5])

                    # Saving Model training results
                    torch.save(model, os.getcwd() + '\\Model Data' + '\\ANN_' + str(ANN_n) + '_Var_' + str(n) + '.pt')
                    '''
                    if n == 63:  # To check error behaviour by variable
                        print('\n', Scores[:, n * 6])
                        print(Scores[:, n * 6 + 3])
                        print(output_data_max[n])
                        print(output_test_norm_np[:, n])
                        print(ypred_model)
                        quit()
                    '''

                else:
                    # Temporal matrix which saves the scores of the Cross Validations for the ANN

                    # For the training set

                    Scores_ANN[cont, n * 6 + 0] = abs(
                        sk.metrics.mean_squared_error(output_data_np[:, n], ytrain_model)) ** (1 / 2)
                    Scores_ANN[cont, n * 6 + 1] = abs(
                        sk.metrics.mean_absolute_error(output_data_np[:, n], ytrain_model))
                    Scores_ANN[cont, n * 6 + 2] = abs(sk.metrics.r2_score(output_data_np[:, n], ytrain_model))

                    # And the test set
                    Scores_ANN[cont, n * 6 + 3] = abs(
                        sk.metrics.mean_squared_error(output_test_norm_np[:, n], ypred_model)) ** (1 / 2)
                    Scores_ANN[cont, n * 6 + 4] = abs(
                        sk.metrics.mean_absolute_error(output_test_norm_np[:, n], ypred_model))
                    Scores_ANN[cont, n * 6 + 5] = abs(sk.metrics.r2_score(output_test_norm_np[:, n], ypred_model))
                    '''
                    Scores_ANN[cont, n * 6 + 0] = (sum(output_data_np[:, n] - ytrain_model) ** 2 / len(output_data_np)) ** (1/2)
                    Scores_ANN[cont, n * 6 + 1] = (sum(abs(output_data_np[:, n] - ytrain_model))/len(output_data_np))
                    Scores_ANN[cont, n * 6 + 2] = abs(sk.metrics.r2_score(output_data_np[:, n], ytrain_model))

                    # And the test set
                    Scores_ANN[cont, n * 6 + 3] = (sum(output_test_norm_np[:, n] - ypred_model) ** 2 / len(output_test_norm_np)) ** (1 / 2)
                    Scores_ANN[cont, n * 6 + 4] = (sum(abs(output_test_norm_np[:, n] - ypred_model))/len(output_test_norm_np))
                    Scores_ANN[cont, n * 6 + 5] = abs(sk.metrics.r2_score(output_test_norm_np[:, n], ypred_model))
                    '''
                del model  # Deleting model to avoid errors in future training

        Times_ANN_times[ANN_n] = time.time() - Time_Dummy_ANN
        Times_ANN_names[ANN_n] = ('ANN ' + str(ANN_n))
        Time_Dummy_ANN = time.time()

    # Sending results to Tiresias_Models datasheet
    Tiresias_Models.sheets['ANN'].range(6, 1).value = np.reshape(name_output, (output_n, -1))
    Tiresias_Models.sheets['ANN'].range(1, 2).value = n_folds
    Tiresias_Models.sheets['ANN'].range(6, 2).value = Results_ANN

    for n in range(len(output_data_max)):
        Tiresias_Models.sheets['ANN'].range(6 + n, 108).value = output_data_max[n]

    Tiresias_Models.save()

    sys.stdout.write("\r\rThe %i ANNs have been trained!\033[K\n\n" % ANN_Tot)
    sys.stdout.flush()

# KRIGING

if K_yn == 1:

    # Model definition - Original Tiresias with multiple outputs, anyway function still works.

    Time_Dummy_K = time.time()
    count_diverge_K = 0
    count_converge_K = 0
    count_max_K = 0

    K_Tot = 1  # Number of Kriging kernels to test
    Times_K_names = np.zeros(K_Tot, dtype=object)  # Array containing the execution time for each ANN
    Times_K_times = np.zeros(K_Tot, dtype=float)  # Array containing the execution time for each ANN
    Scores_K = np.zeros([n_folds, 6 * output_n], dtype=float)  # Finds the overall metrics

    total_K_metrics = n_metrics_columns * K_Tot  # Sheet -Results- in Tiresias_Models total number of columns
    Results_K = np.zeros([output_n, total_K_metrics], dtype=float)  # Matrix displayed in -Results- with all metrics

    print("The Kriging kernels are being trained...")
    rows, cols = input_data_norm.shape
    rows = rows - rows % n_folds
    fold_size = int(rows / n_folds)

    for K_n in range(K_Tot):

        for cont in range(n_folds + 1):

            # Manual division of the data set (for Cross Validation)

            if cont != n_folds:
                # Data is divided in n k-folds. Each fold used once as test set and n-1 as training set.
                # input_data and output_data are the n-1 k-folds used to train the C.V. fold.
                # input_test and output_test are the data of the fold used to test the C.V. training
                input_test_norm = input_data_norm[cont * fold_size:(cont + 1) * fold_size, :]
                output_test_norm = output_data_norm[cont * fold_size:(cont + 1) * fold_size, :]
                input_dummy_0 = input_data_norm[0:cont * fold_size, :]
                input_dummy_1 = input_data_norm[(cont + 1) * fold_size:rows, :]
                output_dummy_0 = output_data_norm[0:cont * fold_size, :]
                output_dummy_1 = output_data_norm[(cont + 1) * fold_size:rows, :]
                input_data = torch.cat((input_dummy_0, input_dummy_1))
                output_data = torch.cat((output_dummy_0, output_dummy_1))

            else:
                # Once C.V. done, use the whole dataset to train the ANN (improve future predictions)
                input_data = input_data_norm
                output_data = output_data_norm

            # TRAINING OF THE ACTUAL ANN (Iterative procedure)

            for n in range(output_n):

                # Re-initializing parameters for second iteration

                learning_rate = 0.1
                model_threshold = 1e-3

                last_loss = 100
                trigger_times = 0
                trigger_no_change = 0
                model_count = 1
                model_max = 1000
                patience = 10
                _ = 0

                likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1)
                model = MultitaskGPModel(input_data_norm, output_data_norm[:, n], likelihood)

                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                loss = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

                model.train()
                likelihood.train()

                # Training of the Kriging model
                while _ < 1:
                    optimizer.zero_grad()  # Cleans gradient so no accumulation occurs
                    # Prediction
                    output_pred = model(input_data_norm)
                    # Loss
                    l = -loss(output_pred, torch.unsqueeze(output_data_norm[:, n], 1))
                    # Gradients
                    l.backward()
                    # Update weights
                    optimizer.step()  # Updating optimizer parameters so gradients decrease (SGD)
                    model_count += 1
                    if model_count % 1 == 0:
                        if cont != n_folds:
                            sys.stdout.write(
                                "\rKernel %i of %i Cross validation in progress (%i/%i)... Variable %s... Actual error = %f (Iteration %i)\033[K"
                                % (K_n + 1, K_Tot, cont + 1, n_folds, name_output[n], l, model_count))
                            sys.stdout.flush()
                        else:
                            sys.stdout.write(
                                "\rKernel %i of %i parameters evaluation in progress... Variable %s... Actual error = %f (Iteration %i)\033[K" %
                                (K_n + 1, K_Tot, name_output[n], l, model_count))
                            sys.stdout.flush()
                    if model_count >= model_max:
                        _ = 1
                        count_max_K += 1
                        if l > last_loss:
                            trigger_times += 1
                    # Early stopping function to avoid long non - convergence
                    if (abs(l - last_loss)) <= model_threshold:
                        trigger_no_change += (patience / 3) + 1
                    if trigger_times >= patience:
                        _ = 1
                        count_diverge_K += 1
                    if trigger_no_change >= patience:
                        _ = 1
                        count_converge_K += 1
                    last_loss = l

                # Testing Kriging

                model.eval()
                likelihood.eval()

                with torch.no_grad():

                    if cont != n_folds:
                        trained_dist = likelihood(model(input_test_norm))
                        ypred_model = np.array(trained_dist.mean)
                        ypred_model = np.ndarray.flatten(ypred_model)

                        train_dist = likelihood(model(input_data))
                        ytrain_model = np.array(train_dist.mean)
                        ytrain_model = np.ndarray.flatten(ytrain_model)

                    output_test_norm_np = np.array(output_test_norm)
                    output_data_np = np.array(output_data)

                if cont == n_folds:
                    # When Cross Validation is done, the metrics are obtained
                    Results_K[n, K_n * n_metrics_columns + 0] = np.mean(Scores_K[:, n * 6 + 0])
                    Results_K[n, K_n * n_metrics_columns + 1] = np.std(Scores_K[:, n * 6 + 0])
                    Results_K[n, K_n * n_metrics_columns + 2] = np.mean(Scores_K[:, n * 6 + 1])
                    Results_K[n, K_n * n_metrics_columns + 3] = np.std(Scores_K[:, n * 6 + 1])
                    Results_K[n, K_n * n_metrics_columns + 4] = np.mean(Scores_K[:, n * 6 + 2])
                    Results_K[n, K_n * n_metrics_columns + 5] = np.mean(Scores_K[:, n * 6 + 3])
                    Results_K[n, K_n * n_metrics_columns + 6] = np.std(Scores_K[:, n * 6 + 3])
                    Results_K[n, K_n * n_metrics_columns + 7] = np.mean(Scores_K[:, n * 6 + 4])
                    Results_K[n, K_n * n_metrics_columns + 8] = np.std(Scores_K[:, n * 6 + 4])
                    Results_K[n, K_n * n_metrics_columns + 9] = np.mean(Scores_K[:, n * 6 + 5])

                    # Saving Model training results
                    torch.save((model, likelihood),
                               os.getcwd() + '\\Model Data' + '\\K_' + str(K_n) + '_Var_' + str(n) + '.pt')
                    '''
                    if n == 63:  # To check error behaviour by variable
                        print('\n', Scores[:, n * 6])
                        print(Scores[:, n * 6 + 3])
                        print(output_data_max[n])
                        print(output_test_norm_np[:, n])
                        print(ypred_model)
                        quit()
                    '''

                else:
                    # Temporal matrix which saves the scores of the Cross Validations for the ANN

                    # For the training set

                    Scores_K[cont, n * 6 + 0] = abs(
                        sk.metrics.mean_squared_error(output_data_np[:, n], ytrain_model)) ** (
                                                        1 / 2)
                    Scores_K[cont, n * 6 + 1] = abs(sk.metrics.mean_absolute_error(output_data_np[:, n], ytrain_model))
                    Scores_K[cont, n * 6 + 2] = abs(sk.metrics.r2_score(output_data_np[:, n], ytrain_model))

                    # And the test set
                    Scores_K[cont, n * 6 + 3] = abs(
                        sk.metrics.mean_squared_error(output_test_norm_np[:, n], ypred_model)) ** (1 / 2)
                    Scores_K[cont, n * 6 + 4] = abs(
                        sk.metrics.mean_absolute_error(output_test_norm_np[:, n], ypred_model))
                    Scores_K[cont, n * 6 + 5] = abs(sk.metrics.r2_score(output_test_norm_np[:, n], ypred_model))

                    del trained_dist, train_dist, ypred_model, ytrain_model

            del model

        Times_K_times[K_n] = time.time() - Time_Dummy_K
        Times_K_names[K_n] = ('K ' + str(K_n))
        Time_Dummy_K = time.time()

    sys.stdout.write("\rThe %i Kriging kernels have been trained successfully\033[K" % K_Tot)
    sys.stdout.flush()
    print('')

    # EXPORTING RESULTS

    # Sending results to Tiresias_Models datasheet
    Tiresias_Models.sheets['Kriging'].range(6, 1).value = np.reshape(name_output, (output_n, -1))
    Tiresias_Models.sheets['Kriging'].range(1, 2).value = n_folds
    Tiresias_Models.sheets['Kriging'].range(6, 2).value = Results_K

    for n in range(len(output_data_max)):
        Tiresias_Models.sheets['Kriging'].range(6 + n, 18).value = output_data_max[n]

    Tiresias_Models.save()

else:
    Results_K = np.full([output_n, 1 * n_metrics_columns], 1000, dtype=float)
    Tiresias_Models.sheets['Kriging'].range(6, 2).value = Results_K

# PRINTING EXECUTION TIMES AND SIMULATION SUMMARY

Tiresias_Models.sheets['Results'].range(6, 1).value = np.reshape(name_output, (output_n, -1))
Tiresias_Models.sheets['Results'].range(1, 2).value = n_folds

for n in range(len(output_data_max)):
    Tiresias_Models.sheets['Results'].range(6 + n, 18).value = output_data_max[n]

Tiresias_Models.save()

print("\n--- Execution finished after: %s seconds ---" % round(time.time() - start_time))

print("\nSpecific execution times:\n")

if regression_yn == 1:
    print('Regression training times: ', Times_reg, 'seconds\n')

if ANN_yn > 1:

    Times_ANN_names = np.array([Times_ANN_names])  # Adjusting array for printing in table
    Times_ANN_times = np.array([Times_ANN_times])

    print(tabulate(np.concatenate((Times_ANN_names.T, Times_ANN_times.T), axis=1), headers=['ANN', 'Time (s)']))
    if K_yn == 1:
        print('')  # For spacing purposes

if K_yn == 1:
    Times_K_names = np.array([Times_K_names])
    Times_K_times = np.array([Times_K_times])

    print(tabulate(np.concatenate((Times_K_names.T, Times_K_times.T), axis=1), headers=['Kernel', 'Time (s)']))

if ANN_yn > 1 or K_yn == 1:
    print("\nM.L. convergence:\n")

if ANN_yn > 1:
    print(tabulate([[(count_max_ANN + count_converge_ANN + count_diverge_ANN), count_converge_ANN, count_diverge_ANN,
                     count_max_ANN]],
                   headers=['Total ANNs trained', 'Converged', 'Diverged', 'Reached max iterations']))
    print('')

if K_yn == 1:
    print(
        tabulate([[(count_max_K + count_converge_K + count_diverge_K), count_converge_K, count_diverge_K, count_max_K]],
                 headers=['Total Kernels trained', 'Converged', 'Diverged', 'Reached max iterations']))
    print('')

plot_yn = input('Plot Feature analysis results? (y/n): ')

if plot_yn == 'y':
    plt.show()

print("Press Enter to continue...")
end = input()
