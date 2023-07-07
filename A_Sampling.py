import numpy as np
import pandas as pd
import xlwings as xw
import matplotlib.pyplot as plt
from numpy import ndarray
from smt.sampling_methods import LHS, Random
from scipy.stats import qmc, skewnorm
import skopt
import time

from F_DoE_variables import DoE

# Calling the Tiresias_Book workbook where Input and Output variables are defined and bounded
workbook = xw.Book('Tiresias.xlsm')
workbook_2 = xw.Book('Tiresias_Data.xlsx')


class Sampling:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.doe = DoE()                         # Class DoE clusters input information for the design. See F_DoE

        self.input_data = None                   # Pandas DataFrame containing the input data which will be fed to HYSYS
        self.input_data_names = None             # Stores the names of the variables that change (important for DoE)
        self.input_data_names_original = None    # Stores the original input names of the variables
        self.output_data_names = None            # Stores the output names of the variables
        self.data_source = None                  # Stores the source of data (HYSYS or real plant)

        self.results_main = None                 # Auxiliary variable for self.input_data
        self.results_secondary = None            # Auxiliary variable for self.input_data
        self.results = None                      # Auxiliary variable for self.input_data

    def __call__(self, *args, **kwargs):

        start_time = time.time()
        self.read_initial_info()
        self.read_limits()
        self.generate_initial_values()
        self.delete_non_changing_vars()
        self.calculate_aux_values()
        self.call_sampling()
        print("\n--- Execution finished after: %s seconds ---\n" % np.round(time.time() - start_time, decimals=2))
        sampling.plot_results()
        print("\nPress Enter to close...")
        end = input()

    def read_initial_info(self):

        # Extracts DoE information from Tiresias such as number of inputs, outputs and required samples.

        self.doe.n_inputs = int(workbook.sheets['Sheet1'].range('C4').current_region.end('down').row - 3)
        self.doe.n_outputs = int(workbook.sheets['Sheet1'].range('Q4').current_region.end('down').row - 3)
        self.doe.n_samples_main = int(workbook.sheets['Sheet1'].range('M14').value)
        self.doe.n_samples_secondary = int(workbook.sheets['Sheet1'].range('N14').value)
        self.doe.variation = float(workbook.sheets['Sheet1'].range(23, 13).value)  # Variation as % of nominal.
        self.data_source = int(workbook.sheets['Sheet1'].range(2, 12).value)       # Hysys or Real Plant Data

        k_folds = int(workbook.sheets['Sheet1'].range('M24').value)                # Number of folds for C.V.
        workbook_2.sheets['Settings'].range('B7').current_region.clear_contents()  # Clears data that may cause problems
        workbook_2.sheets['Settings'].range('B5').value = k_folds

    def read_limits(self):

        # Reads the variable lower and upper boundaries. Also reads the name of input and output variables

        self.doe.limits = np.zeros([self.doe.n_inputs, 5], dtype=float)
        name_input = [0] * self.doe.n_inputs
        name_output = [0] * self.doe.n_outputs

        for n in range(self.doe.n_inputs):

            name_input[n] = workbook.sheets['Sheet1'].range(n + 4, 3).value + '.' + \
                            workbook.sheets['Sheet1'].range(n + 4, 4).value
            self.doe.limits[n, 0] = float(workbook.sheets['Sheet1'].range(n + 4, 7).value)  # Nominal value of variable
            self.doe.limits[n, 3] = float(workbook.sheets['Sheet1'].range(n + 4, 6).value)  # Type of boundary (User)
            self.doe.limits[n, 4] = float(workbook.sheets['Sheet1'].range(n + 4, 2).value)  # Variable changing or not

            if self.doe.limits[n, 4] == 2:  # If variable not changing, limits are set to nominal.
                self.doe.limits[n, 1] = self.doe.limits[n, 2] = self.doe.limits[n, 0]

            else:
                if self.doe.limits[n, 3] == 2:  # If data comes from User, takes the minimum and maximum from excel
                    self.doe.limits[n, 1] = float(workbook.sheets['Sheet1'].range(n + 4, 8).value)  # Minimum value
                    self.doe.limits[n, 2] = float(workbook.sheets['Sheet1'].range(n + 4, 9).value)  # Maximum value

                else:  # If data comes from Tiresias
                    if self.doe.limits[n, 0] == 0:  # If nominal is zero, define limits as zero also to avoid errors
                        self.doe.limits[n, 1] = self.doe.limits[n, 2] = 0

                    else:  # If nominal not zero, limits based on user-defined variation percentage
                        self.doe.limits[n, 1] = self.doe.limits[n, 0] - abs(self.doe.variation * self.doe.limits[n, 0])
                        self.doe.limits[n, 2] = self.doe.limits[n, 0] + abs(self.doe.variation * self.doe.limits[n, 0])

        for n in range(self.doe.n_outputs):
            name_output[n] = workbook.sheets['Sheet1'].range(n + 4, 17).value + '.' + workbook.sheets['Sheet1'].range(
                n + 4, 18).value

        self.input_data_names_original = name_input
        self.output_data_names = name_output

    def generate_initial_values(self):

        # Defines a first_table composed by the nominal values of the variables.
        # The table has the size of the final DoE table (with n_samples + n_samples_secondary rows)

        first_table = np.zeros((int(self.doe.n_samples_main + self.doe.n_samples_secondary),
                               len(self.input_data_names_original)))

        for n in range(len(self.input_data_names_original)):
            first_table[:, n] = self.doe.limits[n, 0]

        self.input_data = pd.DataFrame(first_table, columns=self.input_data_names_original)

    def delete_non_changing_vars(self):

        # Variables which are not changing are deleted from the DoE.
        # Both for variables selected as not changing or with limits set equal (i.e. recycles and adjusters).

        self.doe.n_inputs_aux = self.doe.n_inputs
        n = 0
        name_input_2 = self.input_data_names_original  # Dummy variable where non-changing variables will be deleted.

        while n < self.doe.n_inputs_aux:
            # If variable not changing, or if limits are equal, removes them from sampling
            if self.doe.limits[n, 4] == 2 or self.doe.limits[n, 1] == self.doe.limits[n, 2]:
                self.doe.limits = np.delete(self.doe.limits, n, 0)
                name_input_2 = np.delete(name_input_2, n, 0)
                n -= 1
                self.doe.n_inputs_aux -= 1

            n += 1

        # IDEA FOR LATER: Variation must be modified if convergence window changes.
        # LATER ALSO: Suggest modifications of user-defined minimum or maximum if narrow convergence window.

        self.input_data_names = name_input_2  # Re-define variables so only changing ones are considered for DoE

        # Printing the input data on screen
        np.set_printoptions(suppress=True)
        print("Input variables:\n")
        frame = {'Variable': pd.Series(self.input_data_names),
                 'Nominal': pd.Series(self.doe.limits[:, 0]),
                 'Minimum': pd.Series(self.doe.limits[:, 1]),
                 'Maximum': pd.Series(self.doe.limits[:, 2])}
        sampling_input = pd.DataFrame(frame)
        print(sampling_input.to_string(index=False))

    def get_sample(self, sampling_method, n_samples, sampling_limits, normal_boundaries):

        # Returns a dataset with the DoE given the sampling method, number of samples and sampling limits.
        # Executed with call_sampling function

        x = None
        label = None
        data_export = None

        if sampling_method == 1:
            print("Sampling method: Latin Hypercube")
            sampling_lhs = LHS(xlimits=sampling_limits)
            x = sampling_lhs(n_samples)
            label = "Latin Hypercube"
            data_export = pd.DataFrame(x, columns=self.input_data_names)
        elif sampling_method == 2:
            print("Sampling method: Random")
            sampling_rnd = Random(xlimits=sampling_limits)
            x = sampling_rnd(n_samples)
            label = "Random"
            data_export = pd.DataFrame(x, columns=self.input_data_names)
        elif sampling_method == 3:
            print("Sampling method: LHS by skopt")
            sampling_lhs_skopt = skopt.sampler.lhs.Lhs(lhs_type="classic", criterion=None)
            x = np.array(sampling_lhs_skopt.generate(dimensions=sampling_limits, n_samples=n_samples))
            label = "Latin Hypercube (skopt)"
            data_export = pd.DataFrame(x, columns=self.input_data_names)
        elif sampling_method == 4:
            print("Sampling method: Normal Distribution")
            x = skewnorm.ppf(normal_boundaries.lhd, 0, loc=normal_boundaries.mean, scale=normal_boundaries.std_max)
            label = "Normal Distribution"
            data_export = pd.DataFrame(x, columns=self.input_data_names)
        elif sampling_method == 5:
            print("Sampling method: Skewed normal X1")
            x = skewnorm.ppf(normal_boundaries.lhd, normal_boundaries.sk_factor_1, loc=normal_boundaries.mean,
                             scale=normal_boundaries.std_max)
            label = "Skewed normal X1"
            data_export = pd.DataFrame(x, columns=self.input_data_names)
        elif sampling_method == 6:
            print("Sampling method: Skewed normal X2")
            x = skewnorm.ppf(normal_boundaries.lhd, normal_boundaries.sk_factor_2, loc=normal_boundaries.mean,
                             scale=normal_boundaries.std_max)
            label = "Skewed normal X2"
            data_export = pd.DataFrame(x, columns=self.input_data_names)
        elif sampling_method == 7:
            print("Sampling method: Skewed normal X3")
            x = skewnorm.ppf(normal_boundaries.lhd, normal_boundaries.sk_factor_3, loc=normal_boundaries.mean,
                             scale=normal_boundaries.std_max)
            label = "Skewed normal X3"
            data_export = pd.DataFrame(x, columns=self.input_data_names)
        elif sampling_method == 8:
            print("Sampling method: Sobol")
            sampling_sobol = skopt.sampler.Sobol()
            x = np.array(sampling_sobol.generate(dimensions=sampling_limits, n_samples=n_samples))
            label = "Sobol"
            data_export = pd.DataFrame(x, columns=self.input_data_names)
        elif sampling_method == 9:
            print("Sampling method: Centered LHS")
            sampling_lhs_center = skopt.sampler.lhs.Lhs(lhs_type="centered", criterion=None)
            x = np.array(sampling_lhs_center.generate(dimensions=sampling_limits, n_samples=n_samples))
            label = "Centered LHS"
            data_export = pd.DataFrame(x, columns=self.input_data_names)
        elif sampling_method == 10:
            print("Sampling method: Maximin optimized LHS")
            sampling_lhs_maximin = skopt.sampler.lhs.Lhs(criterion="maximin", iterations=10000)
            x = np.array(sampling_lhs_maximin.generate(dimensions=sampling_limits, n_samples=n_samples))
            label = "Maximin optimized LHS"
            data_export = pd.DataFrame(x, columns=self.input_data_names)
        elif sampling_method == 11:
            print("Sampling method: Correlation optimized LHS")
            sampling_lhs_correlation = skopt.sampler.lhs.Lhs(criterion="correlation", iterations=10000)
            x = np.array(
                sampling_lhs_correlation.generate(dimensions=sampling_limits, n_samples=n_samples))
            label = "Correlation optimized LHS"
            data_export = pd.DataFrame(x, columns=self.input_data_names)
        elif sampling_method == 12:
            print("Sampling method: Ratio optimized LHS")
            sampling_lhs_ratio = skopt.sampler.lhs.Lhs(criterion="ratio", iterations=10000)
            x = np.array(sampling_lhs_ratio.generate(dimensions=sampling_limits, n_samples=n_samples))
            label = "Ratio optimized LHS"
            data_export = pd.DataFrame(x, columns=self.input_data_names)
        elif sampling_method == 13:
            print("Sampling method: Halton")
            sampling_halton = skopt.sampler.Halton()
            x = np.array(sampling_halton.generate(dimensions=sampling_limits, n_samples=n_samples))
            label = "Halton"
            data_export = pd.DataFrame(x, columns=self.input_data_names)
        elif sampling_method == 14:
            print("Sampling method: Hammersly")
            sampling_hammersly = skopt.sampler.Hammersly()
            x = np.array(sampling_hammersly.generate(dimensions=sampling_limits, n_samples=n_samples))
            label = "Hammersly"
            data_export = pd.DataFrame(x, columns=self.input_data_names)
        elif sampling_method == 15:
            print("Sampling method: Grid")
            sampling_grid = skopt.sampler.Grid(border="include", use_full_layout=False)
            x = np.array(sampling_grid.generate(dimensions=sampling_limits, n_samples=n_samples))
            label = "Grid"
            data_export = pd.DataFrame(x, columns=self.input_data_names)

        print(n_samples, "samples generated\n")

        class Results:
            def __init__(self):
                self.x = None
                self.label = None
                self.data = None
                self.norm_data = None
                self.discrepancy = None

        results = Results()
        results.x = x
        results.label = label
        results.data = data_export

        return results

    def calculate_aux_values(self):

        # Calculates some auxiliary variables for the normally-distributed sampling methods (mean, std, skewness, etc)

        class NormalBoundaries:
            def __init__(self):
                self.lhd = None
                self.mean = None
                self.std_max = None
                self.sk_factor_1 = None
                self.sk_factor_2 = None
                self.sk_factor_3 = None

        print("\nDesign of Experiments (DoE) in progress...\n")

        mean = np.zeros([len(self.input_data_names)], dtype=float)
        std_min = np.zeros([len(self.input_data_names)], dtype=float)
        std_max = np.zeros([len(self.input_data_names)], dtype=float)
        dif_min = np.zeros([len(self.input_data_names)], dtype=float)
        dif_max = np.zeros([len(self.input_data_names)], dtype=float)
        bound_low = np.zeros([len(self.input_data_names)], dtype=float)
        bound_high = np.zeros([len(self.input_data_names)], dtype=float)
        skew_factor_1: ndarray = np.zeros([len(self.input_data_names)], dtype=float)
        skew_factor_2: ndarray = np.zeros([len(self.input_data_names)], dtype=float)
        skew_factor_3: ndarray = np.zeros([len(self.input_data_names)], dtype=float)

        # Defining the skewness factor for Normally distributed methods
        for n in range(self.doe.n_inputs_aux):
            mean[n] = self.doe.limits[n, 0]
            dif_min[n] = abs(
                self.doe.limits[n, 0] - self.doe.limits[n, 1])  # Finds absolute difference with minimum limit
            dif_max[n] = abs(
                self.doe.limits[n, 0] - self.doe.limits[n, 2])  # Finds absolute difference with maximum limit
            std_min[n] = 0.341 * min(dif_min[n], dif_max[n])  # Smallest STD obtained via -Empirical rule-
            std_max[n] = 0.341 * max(dif_min[n], dif_max[n])  # Biggest STD obtained via -Empirical rule-

        for n in range(self.doe.n_inputs_aux):

            if dif_min[n] > dif_max[n]:
                bound_low[n] = mean[n] - 1.5 * std_max[n]
                bound_high[n] = mean[n] + 1.5 * std_min[n]
            else:
                bound_low[n] = mean[n] - 1.5 * std_min[n]
                bound_high[n] = mean[n] + 1.5 * std_max[n]

            if std_max[n] == 0:
                skew_factor_1[n] = 0
                skew_factor_2[n] = 0
                skew_factor_3[n] = 0
            else:
                skew_factor_1[n] = ((bound_low[n] - mean[n]) ** 1 + (bound_high[n] - mean[n]) ** 1) / (std_max[n] ** 1)
                skew_factor_2[n] = (-(bound_low[n] - mean[n]) ** 2 + (bound_high[n] - mean[n]) ** 2) / (std_max[n] ** 2)
                skew_factor_3[n] = ((bound_low[n] - mean[n]) ** 3 + (bound_high[n] - mean[n]) ** 3) / (std_max[n] ** 3)

        self.doe.method_main = int(workbook.sheets['Sheet1'].range('M30').value)  # LH, Normal, FF or RND (ComboBox)
        self.doe.method_secondary = int(
            workbook.sheets['Sheet1'].range('N30').value)  # None, LH, Normal, FF or RND (ComboBox)

        lhd = qmc.LatinHypercube(d=self.doe.n_inputs_aux).random(n=self.doe.n_samples_main)
        lhd2 = qmc.LatinHypercube(d=self.doe.n_inputs_aux).random(n=self.doe.n_samples_secondary)

        self.doe.normal_boundaries_main = NormalBoundaries()
        self.doe.normal_boundaries_main.lhd = lhd
        self.doe.normal_boundaries_main.mean = mean
        self.doe.normal_boundaries_main.std_max = std_max
        self.doe.normal_boundaries_main.sk_factor_1 = skew_factor_1
        self.doe.normal_boundaries_main.sk_factor_2 = skew_factor_2
        self.doe.normal_boundaries_main.sk_factor_3 = skew_factor_3

        self.doe.normal_boundaries_secondary = NormalBoundaries()
        self.doe.normal_boundaries_secondary.lhd = lhd2
        self.doe.normal_boundaries_secondary.mean = mean
        self.doe.normal_boundaries_secondary.std_max = std_max
        self.doe.normal_boundaries_secondary.sk_factor_1 = skew_factor_1
        self.doe.normal_boundaries_secondary.sk_factor_2 = skew_factor_2
        self.doe.normal_boundaries_secondary.sk_factor_3 = skew_factor_3

    def call_sampling(self):

        # Calls get_sample function given the requirements of the DoE and prints the sampling results in Tiresias_Data

        self.results_main = self.get_sample(self.doe.method_main, self.doe.n_samples_main,
                                            self.doe.limits[:, [1, 2]],
                                            self.doe.normal_boundaries_main)

        self.results = self.results_main  # To define Results as object

        if self.doe.method_secondary == 1:

            lower_lim = np.min(
                np.vstack((self.results_main.data.min().to_numpy() - 1e-8, self.doe.limits[:, 1] - 1e-8)),
                axis=0)  # Small additions to avoid values on boundary which qmc won't accept
            higher_lim = np.max(
                np.vstack((self.results_main.data.max().to_numpy() + 1e-8, self.doe.limits[:, 2] + 1e-8)),
                axis=0)

            self.results_main.norm_data = qmc.scale(self.results_main.data.to_numpy(), lower_lim, higher_lim,
                                                    reverse=True)

            self.results_main.discrepancy = np.round(qmc.discrepancy(self.results_main.norm_data, method='MD'),
                                                     decimals=5)

            print('Mixture Discrepancy (MD) of ' + self.results_main.label + ' sampling method (main): ',
                  self.results_main.discrepancy)

        else:

            self.doe.method_secondary -= 1
            self.results_secondary = self.get_sample(self.doe.method_secondary, self.doe.n_samples_secondary,
                                                     self.doe.limits[:, [1, 2]], self.doe.normal_boundaries_secondary)
            lower_lim = np.min(
                np.vstack((self.results_main.data.min().to_numpy() - 1e-8,
                           self.results_secondary.data.min().to_numpy() - 1e-8, self.doe.limits[:, 1] - 1e-8)), axis=0)
            higher_lim = np.max(
                np.vstack((self.results_main.data.max().to_numpy() + 1e-8,
                           self.results_secondary.data.max().to_numpy() + 1e-8, self.doe.limits[:, 2] + 1e-8)), axis=0)

            # Primary method normalization and discrepancy

            self.results_main.norm_data = qmc.scale(self.results_main.data.to_numpy(), lower_lim, higher_lim,
                                                    reverse=True)

            self.results_main.discrepancy = np.round(qmc.discrepancy(self.results_main.norm_data, method='MD'),
                                                     decimals=5)

            print('Mixture Discrepancy (MD) of ' + self.results_main.label + ' sampling method (main): ',
                  self.results_main.discrepancy)

            # Primary method normalization and discrepancy

            self.results_secondary.norm_data = qmc.scale(self.results_secondary.data, lower_lim, higher_lim,
                                                         reverse=True)
            self.results_secondary.discrepancy = np.round(qmc.discrepancy(self.results_secondary.norm_data,
                                                                          method='MD'), decimals=5)
            print('Mixture Discrepancy (MD) of ' + self.results_secondary.label + ' sampling method (secondary): ',
                  self.results_secondary.discrepancy)

            self.results.data = pd.concat([self.results_main.data, self.results_secondary.data])
            self.results.norm_data = qmc.scale(self.results.data, lower_lim, higher_lim, reverse=True)
            print(self.results.discrepancy)
            self.results.discrepancy = np.round(qmc.discrepancy(self.results.norm_data, method='MD'), decimals=5)
            print('Overall Mixture Discrepancy (MD) of the sampling: ', self.results.discrepancy)

            ''' # Other Discrepancy measurements #
                'CD' : Centered L2
                'WD' : Wrap around discrepancy
                'L2-star' : L2 discrepancy
            '''

        workbook.sheets['Predictions_Table'].range(1, 1).value = self.results.data

        # In the original table, replaces the non-constant columns only with the DoE done
        for column in self.results.data.columns:
            self.input_data[column] = self.results.data[column]

        workbook_2.sheets['Sheet1'].range(1, 1).value = self.input_data
        workbook_2.sheets['Settings'].range('B7').value = self.results.data.columns.values.tolist()

        for n in range(len(self.output_data_names)):
            workbook_2.sheets['Sheet1'].range(1, n + 2 + len(self.input_data_names_original)).value = \
                self.output_data_names[n]

        workbook_2.sheets['Settings'].range('B2').value = len(self.input_data_names_original)
        workbook_2.sheets['Settings'].range('B3').value = len(self.output_data_names)

        workbook_2.save()

    def plot_results(self):

        # Plotting zone - Plots the sampling points of the first two variables (2D plot)
        plot_yn = input('Plot the sampling results? (y/n): ')

        if plot_yn == 'y':

            # LATER: 3D plot would be good also

            # Which variable will be plotted
            temp1, temp2 = self.doe.limits.shape
            var1 = 0
            if temp1 > 1:
                var2 = 1
            else:
                var2 = 0

            font = {'family': 'Times New Roman',
                    'size': 18}
            plt.rc('font', **font)
            plt.rcParams['figure.constrained_layout.use'] = True

            # plt.title("Design of experiments")
            plt.xlim([self.doe.limits[var1, 1] - abs(0.1 * self.doe.limits[var1, 0]), self.doe.limits[var1, 2] +
                      abs(0.1 * self.doe.limits[var1, 0])])
            plt.ylim([self.doe.limits[var2, 1] - abs(0.1 * self.doe.limits[var2, 0]), self.doe.limits[var2, 2] +
                      abs(0.1 * self.doe.limits[var2, 0])])
            plt.xlabel(self.input_data_names[var1])
            plt.ylabel(self.input_data_names[var2])
            plt.plot(self.results_main.x[:, var1], self.results_main.x[:, var2], "o", label=self.results_main.label)
            if self.results_secondary is not None:
                plt.plot(self.results_secondary.x[:, var1], self.results_secondary.x[:, var2], "o",
                         label=self.results_secondary.label)

            plt.axvline(x=self.doe.limits[var1, 1], color='r', linestyle="--", linewidth=0.7)
            plt.axvline(x=self.doe.limits[var1, 2], color='r', linestyle="--", linewidth=0.7)
            plt.axvline(x=self.doe.limits[var1, 0], color='g', linestyle="--", linewidth=0.7)
            plt.axhline(y=self.doe.limits[var2, 1], color='r', linestyle="--", linewidth=0.7)
            plt.axhline(y=self.doe.limits[var2, 2], color='r', linestyle="--", linewidth=0.7)
            plt.axhline(y=self.doe.limits[var2, 0], color='g', linestyle="--", linewidth=0.7)
            plt.legend(ncol=5)
            plt.show()


sampling = Sampling()
sampling()
