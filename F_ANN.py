import os
import sys
import time
import torch
import numpy as np
import sklearn as sk
import torch.nn as nn

from math import ceil
from F_Functions import Net1, Net2, Net3  # Import the ANN definitions


def ann_training(input_dataframe, output_dataframe, n_folds, ann_yn, n_metrics_columns):

    ann_tot = int(ann_yn - 1)  # Number of ANN's to test selected by user
    ann_names = np.zeros(ann_tot, dtype=object)  # Array containing the execution time for each ANN
    ann_times = np.zeros(ann_tot, dtype=float)  # Array containing the execution time for each ANN
    ann_scores = np.zeros([n_folds, 6 * len(output_dataframe.columns)], dtype=float)  # Finds the overall metrics

    total_ann_metrics = n_metrics_columns * ann_tot
    ann_results = np.zeros([len(output_dataframe.columns), total_ann_metrics], dtype=float)  # Matrix in -Results-

    print("The Artificial Neural Networks (ANN's) are being trained...")
    rows, cols = input_dataframe.to_numpy().shape
    rows = rows - rows % n_folds
    fold_size = int(rows / n_folds)

    ann_time_dummy = time.time()
    ann_count_diverge = 0
    ann_count_converge = 0
    ann_count_max = 0

    input_data_norm = torch.tensor(input_dataframe.to_numpy(), dtype=torch.float32)
    output_data_norm = torch.tensor(output_dataframe.to_numpy(), dtype=torch.float32)

    for ANN_n in range(ann_tot):

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

            for n in range(len(output_dataframe.columns)):

                # The model is selected depending on the actual value of the ANN_n counter

                if ANN_n == 0:

                    # ANN 0: Basic linear model ANN (no hidden layers)

                    model = nn.Linear(len(input_dataframe.columns), 1)

                elif ANN_n == 1:

                    # ANN 1: ANN with one hidden layer with same input_size elements

                    model = Net1(input_size=len(input_dataframe.columns), hidden_size=len(input_dataframe.columns), num_classes=1)

                elif ANN_n == 2:

                    # ANN 2: ANN with one hidden layer with twice input_size elements

                    model = Net1(input_size=len(input_dataframe.columns), hidden_size=len(input_dataframe.columns) * 2, num_classes=1)

                elif ANN_n == 3:

                    # ANN 3: ANN with one hidden layer with half input_size elements

                    model = Net1(input_size=len(input_dataframe.columns), hidden_size=ceil(len(input_dataframe.columns) / 2), num_classes=1)

                elif ANN_n == 4:

                    # ANN 4: ANN with two hidden layers with same input_size elements

                    model = Net2(input_size=len(input_dataframe.columns), hidden_size1=len(input_dataframe.columns), hidden_size2=len(input_dataframe.columns), num_classes=1)

                elif ANN_n == 5:

                    # ANN 5: ANN with two hidden layers with decreasing elements

                    model = Net2(input_size=len(input_dataframe.columns), hidden_size1=ceil(len(input_dataframe.columns) / 2), hidden_size2=ceil(len(input_dataframe.columns) / 3),
                                 num_classes=1)

                elif ANN_n == 6:

                    # ANN 6: ANN with two hidden layers with increasing then decreasing number of elements

                    model = Net2(input_size=len(input_dataframe.columns), hidden_size1=len(input_dataframe.columns) * 2, hidden_size2=len(input_dataframe.columns),
                                 num_classes=1)

                elif ANN_n == 7:

                    # ANN 7: ANN with two hidden layers with increasing number of elements

                    model = Net2(input_size=len(input_dataframe.columns), hidden_size1=len(input_dataframe.columns) * 2, hidden_size2=len(input_dataframe.columns) * 3, num_classes=1)

                elif ANN_n == 8:

                    # ANN 8: ANN with two hidden layers with decreasing then increasing number of elements

                    model = Net2(input_size=len(input_dataframe.columns), hidden_size1=ceil(len(input_dataframe.columns) / 2), hidden_size2=len(input_dataframe.columns),
                                 num_classes=1)

                elif ANN_n == 9:

                    # ANN 9: ANN with three hidden layers n_input - 3 - n_input - 3 - n_ output (Encoder - Decoder)

                    model = Net3(input_size=len(input_dataframe.columns), hidden_size1=3, hidden_size2=len(input_dataframe.columns), hidden_size3=3, num_classes=1)

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
                                % (ANN_n + 1, ann_tot, cont + 1, n_folds, output_dataframe.columns[n], l, model_count))
                            sys.stdout.flush()
                        else:
                            sys.stdout.write(
                                "\rANN %i of %i parameters evaluation in progress... Variable %s... Actual error = %f (Iteration %i)\033[K" %
                                (ANN_n + 1, ann_tot, output_dataframe.columns[n], l, model_count))
                            sys.stdout.flush()

                    if model_count >= model_max:
                        _ = 1
                        ann_count_max += 1

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
                        ann_count_diverge += 1
                    if trigger_no_change >= patience:
                        _ = 1
                        ann_count_converge += 1

                    last_loss = l

                model.eval()  # Sets the model in evaluation mode

                with torch.no_grad():

                    if cont != n_folds:
                        ypred_model = np.array(model(input_test_norm))  # Reversing normalization
                        ypred_model = np.ndarray.flatten(ypred_model)
                        y_train_model = np.array(model(input_data))  # Reversing normalization
                        y_train_model = np.ndarray.flatten(y_train_model)

                    output_test_norm_np = np.array(output_test_norm)
                    output_data_np = np.array(output_data)

                if cont == n_folds:
                    # When Cross Validation is done, the metrics are obtained

                    ann_results[n, ANN_n * n_metrics_columns + 0] = np.mean(ann_scores[:, n * 6 + 0])
                    ann_results[n, ANN_n * n_metrics_columns + 1] = np.std(ann_scores[:, n * 6 + 0])
                    ann_results[n, ANN_n * n_metrics_columns + 2] = np.mean(ann_scores[:, n * 6 + 1])
                    ann_results[n, ANN_n * n_metrics_columns + 3] = np.std(ann_scores[:, n * 6 + 1])
                    ann_results[n, ANN_n * n_metrics_columns + 4] = np.mean(ann_scores[:, n * 6 + 2])
                    ann_results[n, ANN_n * n_metrics_columns + 5] = np.mean(ann_scores[:, n * 6 + 3])
                    ann_results[n, ANN_n * n_metrics_columns + 6] = np.std(ann_scores[:, n * 6 + 3])
                    ann_results[n, ANN_n * n_metrics_columns + 7] = np.mean(ann_scores[:, n * 6 + 4])
                    ann_results[n, ANN_n * n_metrics_columns + 8] = np.std(ann_scores[:, n * 6 + 4])
                    ann_results[n, ANN_n * n_metrics_columns + 9] = np.mean(ann_scores[:, n * 6 + 5])

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

                    ann_scores[cont, n * 6 + 0] = abs(
                        sk.metrics.mean_squared_error(output_data_np[:, n], y_train_model)) ** (1 / 2)
                    ann_scores[cont, n * 6 + 1] = abs(sk.metrics.mean_absolute_error(output_data_np[:, n], y_train_model))
                    ann_scores[cont, n * 6 + 2] = abs(sk.metrics.r2_score(output_data_np[:, n], y_train_model))

                    # And the test set
                    ann_scores[cont, n * 6 + 3] = abs(
                        sk.metrics.mean_squared_error(output_test_norm_np[:, n], ypred_model)) ** (1 / 2)
                    ann_scores[cont, n * 6 + 4] = abs(
                        sk.metrics.mean_absolute_error(output_test_norm_np[:, n], ypred_model))
                    ann_scores[cont, n * 6 + 5] = abs(sk.metrics.r2_score(output_test_norm_np[:, n], ypred_model))
                    '''
                    ann_scores[cont, n * 6 + 0] = (sum(output_data_np[:, n] - ytrain_model) ** 2 / len(output_data_np)) ** (1/2)
                    ann_scores[cont, n * 6 + 1] = (sum(abs(output_data_np[:, n] - ytrain_model))/len(output_data_np))
                    ann_scores[cont, n * 6 + 2] = abs(sk.metrics.r2_score(output_data_np[:, n], ytrain_model))
    
                    # And the test set
                    ann_scores[cont, n * 6 + 3] = (sum(output_test_norm_np[:, n] - ypred_model) ** 2 / len(output_test_norm_np)) ** (1 / 2)
                    ann_scores[cont, n * 6 + 4] = (sum(abs(output_test_norm_np[:, n] - ypred_model))/len(output_test_norm_np))
                    ann_scores[cont, n * 6 + 5] = abs(sk.metrics.r2_score(output_test_norm_np[:, n], ypred_model))
                    '''
                del model  # Deleting model to avoid errors in future training

        ann_times[ANN_n] = time.time() - ann_time_dummy
        ann_names[ANN_n] = ('ANN ' + str(ANN_n))
        ann_time_dummy = time.time()

    sys.stdout.write("\r\rThe %i ANNs have been trained!\033[K\n\n" % ann_tot)
    sys.stdout.flush()

    count_ann = [ann_count_converge, ann_count_diverge, ann_count_max]

    return ann_times, ann_names, ann_results, count_ann
