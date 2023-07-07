import os
import time

import numpy as np
import joblib as jl
import pandas as pd
import xlwings as xw
import sklearn as sk
import matplotlib.pyplot as plt

from F_Regression import regression
from F_ANN import ann_training
from F_Kriging import kriging_training

Tiresias_Book = xw.Book('Tiresias.xlsm')
Tiresias_Data = xw.Book('Tiresias_Data.xlsx')
Tiresias_Models = xw.Book('Tiresias_models.xlsx')


class Tiresias:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data_source = None
        self.n_simulations = None
        self.features = None
        self.pca = False
        self.pca_components = None

        self.initial_size = None
        self.input_data = None
        self.input_data_norm = None

        self.output_data = None
        self.output_data_norm = None

        self.dataset = None
        self.dataset_norm = None

    def __call__(self, *args, **kwargs):

        start_time = time.time()
        regression_yn = int(Tiresias_Book.sheets['RUN'].range('E15').value)  # Train regression models y/n
        ann_yn = int(Tiresias_Book.sheets['RUN'].range('E16').value)  # Train ANNs and number of ANNs to train
        kriging_yn = int(Tiresias_Book.sheets['RUN'].range('E17').value)  # Train Kriging model y/n
        k_folds = int(Tiresias_Data.sheets['Settings'].range('B5').value)

        self.read_initial_data()
        self.data_cleaning()
        self.data_normalization()
        self.feature_extraction()

        n_metrics_columns = 10  # RMSE, STD, MAE, STD, R2 Train and Test
        n_reg_models = 9  # Regression models to test
        n_ann_models = 10  # ANNs to test
        n_kriging_models = 1  # Kriging kernels to test

        total_reg_metrics = n_metrics_columns * n_reg_models
        total_ann_metrics = n_metrics_columns * n_ann_models
        total_kriging_metrics = n_metrics_columns * n_kriging_models

        reg_time = 0

        ann_times_table = None
        ann_convergence_table = None

        kriging_times_table = None
        kriging_convergence_table = None

        reg_results = np.full([len(self.output_data.columns), total_reg_metrics], 1000, dtype=float)
        ann_results = np.full([len(self.output_data.columns), total_ann_metrics], 1000, dtype=float)
        kriging_results = np.full([len(self.output_data.columns), total_kriging_metrics], 1000, dtype=float)

        Tiresias_Models.sheets['Regression'].range(6, 2).value = reg_results
        Tiresias_Models.sheets['ANN'].range(6, 2).value = ann_results
        Tiresias_Models.sheets['Kriging'].range(6, 2).value = kriging_results

        Tiresias_Models.sheets['Results'].range(6, 1).value = np.reshape(self.output_data.columns,
                                                                         (len(self.output_data.columns), -1))

        if regression_yn == 1:

            reg_time, reg_results = regression(self.input_data_norm, self.output_data_norm, k_folds,
                                               n_metrics_columns, n_reg_models)

        Tiresias_Models.sheets['Regression'].range(6, 1).value = np.reshape(self.output_data.columns,
                                                                            (len(self.output_data.columns), -1))
        Tiresias_Models.sheets['Regression'].range(1, 2).value = k_folds
        Tiresias_Models.sheets['Regression'].range(6, 2).value = reg_results

        if ann_yn > 1:

            ann_times, ann_names, ann_results, count_ann = ann_training(self.input_data_norm, self.output_data_norm,
                                                                        k_folds, ann_yn, n_metrics_columns)
            ann_times_table = pd.DataFrame({'ANN': ann_names, 'time (s)': ann_times})
            ann_convergence_table = pd.DataFrame({'ANNs': ['Trained', 'Converged', 'Diverged',
                                                           'Max iterations'],
                                                  'Number': [sum(count_ann), count_ann[0], count_ann[1], count_ann[2]]})

        Tiresias_Models.sheets['ANN'].range(6, 1).value = np.reshape(self.output_data.columns,
                                                                     (len(self.output_data.columns), -1))
        Tiresias_Models.sheets['ANN'].range(1, 2).value = k_folds
        Tiresias_Models.sheets['ANN'].range(6, 2).value = ann_results

        Tiresias_Models.save()

        if kriging_yn == 1:
            kriging_times, kriging_names, kriging_results, count_kriging = kriging_training(self.input_data_norm,
                                                                                            self.output_data_norm,
                                                                                            k_folds, kriging_yn,
                                                                                            n_metrics_columns)
            kriging_times_table = pd.DataFrame({'kriging': kriging_names, 'time (s)': kriging_times})
            kriging_convergence_table = pd.DataFrame({'Kriging': ['Trained', 'Converged', 'Diverged',
                                                                  'Max iterations'],
                                                      'Number': [sum(count_kriging), count_kriging[0], count_kriging[1],
                                                                 count_kriging[2]]})

        Tiresias_Models.sheets['Kriging'].range(6, 1).value = np.reshape(self.output_data.columns,
                                                                         (len(self.output_data.columns), -1))
        Tiresias_Models.sheets['Kriging'].range(1, 2).value = k_folds
        Tiresias_Models.sheets['Kriging'].range(6, 2).value = kriging_results

        Tiresias_Models.save()

        print("\n--- Execution finished after: %s seconds ---" % round(time.time() - start_time))

        print("\nRun information:\n")

        if regression_yn == 1:
            print('Regression training times: ', reg_time, 'seconds\n')

        if ann_yn > 1:
            print('ANNs training times: \n')
            print(ann_times_table.to_string(index=False), "\n")
            print('ANNs iterations information: \n')
            print(ann_convergence_table.to_string(index=False), "\n")

        if kriging_yn == 1:
            print('Kriging training times: \n')
            print(kriging_times_table.to_string(index=False), "\n")
            print('Kriging iterations information: \n')
            print(kriging_convergence_table.to_string(index=False), "\n")

        self.fill_predictions_sheet()
        self.plot_pca()

        print("Press Enter to continue...")
        end = input()

    def read_initial_data(self):

        input_n = Tiresias_Data.sheets['Settings'].range('B2').value
        input_names = Tiresias_Data.sheets['Sheet1'].range((1, 1), (1, input_n)).value
        output_n = Tiresias_Data.sheets['Settings'].range('B3').value
        output_names = Tiresias_Data.sheets['Sheet1'].range((1, input_n + 1), (1, input_n + output_n)).value
        real_input_n = int(Tiresias_Data.sheets['Settings'].range('B7').current_region.end('right').column - 1)
        real_input_names = Tiresias_Data.sheets['Settings'].range((7, 2), (7, real_input_n + 1)).value
        names_to_delete = list(input_names)  # List important, otherwise changes occur simultaneously.

        self.features = int(Tiresias_Data.sheets['Settings'].range('B4').value)

        if type(input_names) is str:  # If only one data, the name becomes a str so turn it into list
            input_names = [input_names]

        if type(output_names) is str:
            output_names = [output_names]

        for n in range(len(real_input_names)):
            try:  # Find elements in input_names not in real_names_input. If no coincidence found, pass.
                names_to_delete.remove(real_input_names[n])
            except Exception:
                pass

        self.data_source = int(Tiresias_Book.sheets['Sheet1'].range(2, 12).value)  # Hysys or Real Plant Data
        self.n_simulations = int(Tiresias_Data.sheets['Sheet1'].range('A4').current_region.end('down').row)

        self.input_data = pd.DataFrame(Tiresias_Data.sheets['Sheet1'].
                                       range((2, 1), (self.n_simulations, input_n)).value,
                                       columns=input_names)
        self.output_data = pd.DataFrame(Tiresias_Data.sheets['Sheet1'].
                                        range((2, input_n + 1), (self.n_simulations, input_n + output_n)).value,
                                        columns=output_names)

        self.input_data.drop(columns=names_to_delete, axis=1, inplace=True)  # Replace non-changing variables

        # Tiresias_Data.close()

    def data_cleaning(self):

        self.initial_size = len(self.input_data)
        self.dataset = pd.concat([self.input_data, self.output_data], axis=1)

        print("Data Cleaning in process...")

        if self.data_source == 2:  # If real plant data, deletes empty entries

            # Turn any empty space into NaN to delete it later
            self.dataset = self.dataset[self.dataset != ""]
            self.dataset.dropna(axis=0, how='any', inplace=True)

        else:  # If hysys simulation, deletes empty entries plus non-physically meaningful data

            # Hysys returns -32767 value when non-convergence. We turn these values into NaN
            self.dataset = self.dataset[self.dataset != -32767]

            # Only Temperature and Heat may be negative, so all other negative values are assigned NaN
            for column in self.dataset.columns:
                if column[-1] != "T" and column[-1] != "Q":
                    self.dataset[column] = self.dataset[column].where(self.dataset[column] >= 0)

            # Now we proceed to delete the NaN values
            self.dataset.dropna(axis=0, how='any', inplace=True)

            self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)  # Shuffles data and resets pandas index

        Tiresias_Data.sheets['Pre-processed'].range('A1').value = self.dataset
        self.input_data = self.dataset.iloc[:, :len(self.input_data.columns)]
        self.output_data = self.dataset.iloc[:, len(self.input_data.columns):len(self.input_data.columns) + len(
            self.output_data.columns)]

        print("\nData has ben validated. %i rows were removed." % (self.initial_size - len(self.dataset)))

    def data_normalization(self):

        scaler_input = sk.preprocessing.MinMaxScaler()
        scaler_output = sk.preprocessing.MinMaxScaler()

        self.input_data_norm = scaler_input.fit_transform(self.input_data.to_numpy())
        self.input_data_norm = pd.DataFrame(self.input_data_norm, columns=self.input_data.columns)
        self.output_data_norm = scaler_output.fit_transform(self.output_data.to_numpy())
        self.output_data_norm = pd.DataFrame(self.output_data_norm, columns=self.output_data.columns)

        jl.dump((scaler_input, scaler_output), os.getcwd() + '\\Model Data' + '\\Scaler')  # Saving the scaling info
        jl.dump((self.input_data_norm.columns.tolist(), self.output_data_norm.columns.tolist()),
                os.getcwd() + '\\Model Data' + '\\Variables')

    def feature_extraction(self):

        print("\nEffective number of samples: ", len(self.dataset))
        print("Number of input variables: ", len(self.input_data.columns))

        # Maximum number of features cannot exceed the maximum of number of simulations or input variables
        if self.features > min(len(self.dataset), len(self.input_data.columns)):
            print("Number of features chosen by the user: ", self.features)
            self.features = min(len(self.dataset), len(self.input_data.columns))
            print("The number of features has been set to", self.features,
                  "which is the maximum possible number of features for this case \n")
        else:
            print("Number of features: ", self.features, "\n")

        self.pca_components = sk.decomposition.PCA(n_components=self.features)
        self.pca_components = self.pca_components.fit(self.input_data_norm.to_numpy())

        if self.features != min(len(self.dataset), len(self.input_data.columns)):
            self.pca = True
            input_data_norm_pca = self.pca_components.transform(self.input_data_norm.to_numpy())
            self.input_data_norm = pd.DataFrame(input_data_norm_pca, columns=self.input_data.columns)

        jl.dump((self.pca_components, self.pca), os.getcwd() + '\\Model Data' + '\\PCA')  # Saving transformation data

    def plot_pca(self):

        pca_weight = self.pca_components.components_
        mean_weight = [0] * len(self.input_data.columns)
        names = [0] * len(self.input_data.columns)

        for n in range(len(self.input_data.columns)):
            mean_weight[n] = np.mean(pca_weight[:, n])

        order = np.argsort(mean_weight)

        if len(self.input_data.columns) == 1:
            names = self.input_data.columns
        else:
            for n in range(len(pca_weight[0])):
                names[n] = self.input_data.columns[order[n]]

        font = {'size': 10}
        plt.rc('font', **font)
        plt.rcParams['figure.constrained_layout.use'] = True

        plt.bar(names, mean_weight)
        plt.xticks(rotation=90)
        plt.ylabel('Average PCA weight')

        plot_yn = input('Plot Feature analysis results? (y/n): ')

        if plot_yn == 'y':
            plt.show()

    def fill_predictions_sheet(self):

        Tiresias_Book.sheets['Predictions_Table'].range(1, len(self.input_data.columns) + 1).value = \
            self.output_data.columns.values

        for n in range(len(self.input_data.columns)):
            input_varname_temp = self.input_data.columns[n].split('.')

            Tiresias_Book.sheets['Predictions'].range(n + 4, 1).value = input_varname_temp[0]
            Tiresias_Book.sheets['Predictions'].range(n + 4, 2).value = input_varname_temp[1]
            Tiresias_Book.sheets['Predictions'].range(n + 4, 4).value = 0

            if input_varname_temp[1] == 'T':
                Tiresias_Book.sheets['Predictions'].range(n + 4, 3).value = "C"
            elif input_varname_temp[1] == 'P':
                Tiresias_Book.sheets['Predictions'].range(n + 4, 3).value = "bar"
            elif input_varname_temp[1] == 'F':
                Tiresias_Book.sheets['Predictions'].range(n + 4, 3).value = "kg/h"
            else:
                Tiresias_Book.sheets['Predictions'].range(n + 4, 3).value = "Mol Frac"

        for n in range(len(self.output_data.columns)):
            Tiresias_Book.sheets['Predictions'].range(n + 4, 6).value = \
                Tiresias_Book.sheets['Sheet1'].range(n + 4, 17).value
            Tiresias_Book.sheets['Predictions'].range(n + 4, 7).value = \
                Tiresias_Book.sheets['Sheet1'].range(n + 4, 18).value
            Tiresias_Book.sheets['Predictions'].range(n + 4, 8).value = \
                Tiresias_Book.sheets['Sheet1'].range(n + 4, 19).value


tiresias = Tiresias()
tiresias()
