import os
import torch
import numpy as np
import joblib as jl
import xlwings as xw
import sklearn as sk

Tiresias_Book = xw.Book('Tiresias.xlsm')


class Predictions:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_best = None
        self.input_test = None
        self.input_test_table = None
        self.input_names = None
        self.output_names = None
        self.scaler_output = None

    def __call__(self, *args, **kwargs):

        self.get_data()
        self.fetch_data()
        self.export_data()

    def get_data(self):

        pca_components, pca = jl.load(os.getcwd() + '\\Model Data' + '\\PCA')
        scaler_input, self.scaler_output = jl.load(os.getcwd() + '\\Model Data' + '\\Scaler')
        self.input_names, self.output_names = jl.load(os.getcwd() + '\\Model Data' + '\\Variables')

        self.model_best = Tiresias_Book.sheets['Predictions'].range((4, 9), (4 + len(self.output_names) - 1, 9)).value

        if type(self.model_best) is str:  # If only one output, turn into list
            self.model_best = [self.model_best]

        input_data = Tiresias_Book.sheets['Predictions'].range((4, 4), (4 + len(self.input_names) - 1, 4)).value

        # If no data in cell, consider it a zero to avoid errors in code.

        if input_data is None:
            if len(self.input_names) > 1:
                input_data = [0] * len(self.input_names)
            else:
                input_data = 0.0

        if type(input_data) is float:
            input_data = [[input_data]]
        elif type(input_data[0]) is float:
            input_data = [input_data]

        input_data = scaler_input.transform(input_data)

        n_predictions = int(
            Tiresias_Book.sheets['Predictions_Table'].range('A1').current_region.end('down').row - 1)  # Num of rows

        input_data_table = Tiresias_Book.sheets['Predictions_Table'].range((2, 1), (n_predictions + 1,
                                                                                    len(self.input_names))).value

        if type(input_data_table) is float:
            input_data_table = [[input_data_table]]
        elif input_data_table[0] is None:
            if len(self.input_names) > 1:
                input_data_table = [0] * len(self.input_names)
            else:
                input_data_table = [[0.0]]
        elif type(input_data_table[0]) is float:
            input_data_table = [input_data_table]

        input_data_table = scaler_input.transform(input_data_table)

        # Avoid PCA if Features = max features (known from B_Tiresias)
        if pca:
            self.input_test = pca_components.transform(input_data)
            self.input_test = torch.tensor(self.input_test, dtype=torch.float32)
            self.input_test_table = pca_components.transform(input_data_table)
            self.input_test_table = torch.tensor(self.input_test_table, dtype=torch.float32)
        else:
            self.input_test = torch.tensor(input_data, dtype=torch.float32)
            self.input_test_table = torch.tensor(input_data_table, dtype=torch.float32)

    def fetch_data(self):

        for n in range(len(self.output_names)):

            model_name = self.model_best[n]

            if model_name[0:7] == "Kriging":
                k_n = model_name[-1]  # Takes the last character, which is the number of the best Kernel

                model, likelihood = torch.load(os.getcwd() + '\\Model Data' + '\\K_' + k_n + '_Var_' + str(n) + '.pt')
                model.eval()

                with torch.no_grad():

                    distribution = likelihood(model(torch.unsqueeze(self.input_test, 0)))
                    prediction = self.scaler_output.inverse_transform(distribution.mean)  # Un-do the normalization
                    Tiresias_Book.sheets['Predictions'].range(4 + n, 10).value = prediction

                    distribution_table = likelihood(model(self.input_test_table))
                    prediction_table = self.scaler_output.inverse_transform(distribution_table.mean)
                    prediction_table = prediction_table.tolist()
                    Tiresias_Book.sheets['Predictions_Table'].range(2, len(self.input_names) + n + 1).value = \
                        prediction_table

            elif model_name[0:3] == "ANN":

                ann_n = model_name[-1]  # Takes the last character, which is the number of the best ANN

                model = torch.load(os.getcwd() + '\\Model Data' + '\\ANN_' + ann_n + '_Var_' + str(n) + '.pt')
                model.eval()

                with torch.no_grad():
                    prediction = self.scaler_output.inverse_transform(model(self.input_test))
                    Tiresias_Book.sheets['Predictions'].range(4 + n, 10).value = prediction

                    prediction_table = self.scaler_output.inverse_transform(model(self.input_test_table))
                    prediction_table = prediction_table.tolist()
                    Tiresias_Book.sheets['Predictions_Table'].range(2, len(self.input_names) + n + 1).value = \
                        prediction_table

            else:

                x_test = np.array(self.input_test)
                x_test_table = np.array(self.input_test_table)

                if self.model_best[n] == "Second Order Polynomial Regression":
                    poly = sk.preprocessing.PolynomialFeatures(degree=2)
                    x_test = poly.fit_transform(x_test)
                    x_test_table = poly.fit_transform(x_test_table)
                elif self.model_best[n] == "Third Order Polynomial Regression":
                    poly = sk.preprocessing.PolynomialFeatures(degree=3)
                    x_test = poly.fit_transform(x_test)
                    x_test_table = poly.fit_transform(x_test_table)
                elif self.model_best[n] == "Fourth Order Polynomial Regression":
                    poly = sk.preprocessing.PolynomialFeatures(degree=4)
                    x_test = poly.fit_transform(x_test)
                    x_test_table = poly.fit_transform(x_test_table)

                model = jl.load(os.getcwd() + '\\Model Data' + '\\' + self.model_best[n] + str(n))

                y_test = model.predict(x_test)

                Tiresias_Book.sheets['Predictions'].range(4 + n, 10).value = np.round(y_test, decimals=3)

                y_test_table = torch.tensor(model.predict(x_test_table), dtype=torch.float32)
                prediction = torch.unsqueeze(y_test_table, 1)
                prediction = prediction.tolist()
                Tiresias_Book.sheets['Predictions_Table'].range(2, len(self.input_names) + n + 1).value = prediction

    def export_data(self):

        output_data = Tiresias_Book.sheets['Predictions'].range((4, 10), (4 + len(self.output_names) - 1, 10)).value


        if type(output_data) is float:
            output_data = [[output_data]]
        elif type(output_data[0]) is float:
            output_data = [output_data]

        output_data = self.scaler_output.inverse_transform(output_data)

        for n in range(len(self.output_names)):
            Tiresias_Book.sheets['Predictions'].range(4 + n, 10).value = output_data[0, n]

        output_data_table = Tiresias_Book.sheets['Predictions_Table'].range((2, len(self.input_names) + 1),
                                                                            (len(self.input_test_table) + 1,
                                                                             len(self.input_names) +
                                                                             len(self.output_names))).value

        if type(output_data_table) is float:
            output_data_table = [output_data_table]
        if type(output_data_table[0]) is float:
            output_data_table = np.reshape(np.array(output_data_table), (-1, 1))

        output_data_table = self.scaler_output.inverse_transform(output_data_table)
        Tiresias_Book.sheets['Predictions_Table'].range(2, len(self.input_names) + 1).value = output_data_table


predictions = Predictions()
predictions()
