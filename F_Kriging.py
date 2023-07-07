import os
import sys
import time
import torch
import gpytorch
import numpy as np
import sklearn as sk

from F_Functions import MultitaskGPModel  # Import the ANN and Kriging definitions


def kriging_training(input_dataframe, output_dataframe, n_folds, k_yn, n_metrics_columns):

    k_tot = k_yn  # Number of Kriging kernels to test. Change as ann in case kernels are added.
    k_names = np.zeros(k_tot, dtype=object)  # Array containing the execution time for each ANN
    k_times = np.zeros(k_tot, dtype=float)  # Array containing the execution time for each ANN
    k_scores = np.zeros([n_folds, 6 * len(output_dataframe.columns)], dtype=float)  # Finds the overall metrics

    total_k_metrics = n_metrics_columns * k_tot
    k_results = np.zeros([len(output_dataframe.columns), total_k_metrics], dtype=float)  # Matrix displayed in -Results-

    print("The Kriging kernels are being trained...")
    rows, cols = input_dataframe.to_numpy().shape
    rows = rows - rows % n_folds
    fold_size = int(rows/n_folds)

    k_time_dummy = time.time()
    k_count_diverge = 0
    k_count_converge = 0
    k_count_max = 0

    input_data_norm = torch.tensor(input_dataframe.to_numpy(), dtype=torch.float32)
    output_data_norm = torch.tensor(output_dataframe.to_numpy(), dtype=torch.float32)

    for K_n in range(k_tot):

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
                                % (K_n + 1, k_tot, cont + 1, n_folds, output_dataframe.columns[n], l, model_count))
                            sys.stdout.flush()
                        else:
                            sys.stdout.write(
                                "\rKernel %i of %i parameters evaluation in progress... Variable %s... Actual error = %f (Iteration %i)\033[K" %
                                (K_n + 1, k_tot, output_dataframe.columns[n], l, model_count))
                            sys.stdout.flush()
                    if model_count >= model_max:
                        _ = 1
                        k_count_max += 1
                        if l > last_loss:
                            trigger_times += 1
                    # Early stopping function to avoid long non - convergence
                    if (abs(l - last_loss)) <= model_threshold:
                        trigger_no_change += (patience / 3) + 1
                    if trigger_times >= patience:
                        _ = 1
                        k_count_diverge += 1
                    if trigger_no_change >= patience:
                        _ = 1
                        k_count_converge += 1
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
                        y_train_model = np.array(train_dist.mean)
                        y_train_model = np.ndarray.flatten(y_train_model)

                    output_test_norm_np = np.array(output_test_norm)
                    output_data_np = np.array(output_data)

                if cont == n_folds:
                    # When Cross Validation is done, the metrics are obtained
                    k_results[n, K_n * n_metrics_columns + 0] = np.mean(k_scores[:, n * 6 + 0])
                    k_results[n, K_n * n_metrics_columns + 1] = np.std(k_scores[:, n * 6 + 0])
                    k_results[n, K_n * n_metrics_columns + 2] = np.mean(k_scores[:, n * 6 + 1])
                    k_results[n, K_n * n_metrics_columns + 3] = np.std(k_scores[:, n * 6 + 1])
                    k_results[n, K_n * n_metrics_columns + 4] = np.mean(k_scores[:, n * 6 + 2])
                    k_results[n, K_n * n_metrics_columns + 5] = np.mean(k_scores[:, n * 6 + 3])
                    k_results[n, K_n * n_metrics_columns + 6] = np.std(k_scores[:, n * 6 + 3])
                    k_results[n, K_n * n_metrics_columns + 7] = np.mean(k_scores[:, n * 6 + 4])
                    k_results[n, K_n * n_metrics_columns + 8] = np.std(k_scores[:, n * 6 + 4])
                    k_results[n, K_n * n_metrics_columns + 9] = np.mean(k_scores[:, n * 6 + 5])

                    # Saving Model training results
                    torch.save((model, likelihood), os.getcwd() + '\\Model Data' + '\\K_' + str(K_n) + '_Var_' + str(n) + '.pt')
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

                    k_scores[cont, n * 6 + 0] = abs(sk.metrics.mean_squared_error(output_data_np[:, n], y_train_model)) ** (
                                1 / 2)
                    k_scores[cont, n * 6 + 1] = abs(sk.metrics.mean_absolute_error(output_data_np[:, n], y_train_model))
                    k_scores[cont, n * 6 + 2] = abs(sk.metrics.r2_score(output_data_np[:, n], y_train_model))

                    # And the test set
                    k_scores[cont, n * 6 + 3] = abs(
                        sk.metrics.mean_squared_error(output_test_norm_np[:, n], ypred_model)) ** (1 / 2)
                    k_scores[cont, n * 6 + 4] = abs(sk.metrics.mean_absolute_error(output_test_norm_np[:, n], ypred_model))
                    k_scores[cont, n * 6 + 5] = abs(sk.metrics.r2_score(output_test_norm_np[:, n], ypred_model))

                    del trained_dist, train_dist, ypred_model, y_train_model

            del model

        k_times[K_n] = time.time() - k_time_dummy
        k_names[K_n] = ('K ' + str(K_n))
        k_time_dummy = time.time()

    sys.stdout.write("\rThe %i Kriging kernels have been trained successfully\033[K" % k_tot)
    sys.stdout.flush()

    count_k = [k_count_converge, k_count_diverge, k_count_max]

    return k_times, k_names, k_results, count_k
