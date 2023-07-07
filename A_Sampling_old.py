import numpy as np
import xlwings as xw
import matplotlib.pyplot as plt
from numpy import ndarray
from tabulate import tabulate
from smt.sampling_methods import LHS, Random, FullFactorial
from scipy.stats import qmc, skewnorm
import skopt
import time

# Calling the Tiresias_Book workbook where Input and Output variables are defined and bounded
workbook = xw.Book('Tiresias_Book.xlsm')

# Calling the number of Input and output variables from the Worksheet "Sheet1"
start_time = time.time()
n_input = int(workbook.sheets['Sheet1'].range('C4').current_region.end('down').row - 3)
n_output = int(workbook.sheets['Sheet1'].range('Q4').current_region.end('down').row - 3)
n_simulations = int(workbook.sheets['Sheet1'].range('M14').value)  # User defined
n_simulations_2 = int(workbook.sheets['Sheet1'].range('N14').value)  # User defined
sampling_method = int(workbook.sheets['Sheet1'].range('M30').value)   # LH, Normal, FF or RND (ComboBox)
sampling_method_2 = int(workbook.sheets['Sheet1'].range('N30').value)   # None, LH, Normal, FF or RND (ComboBox)
data_source = int(workbook.sheets['Sheet1'].range(2, 12).value)  # Hysys or Real Plant Data

# Making initial necessary variable initializations for the code to run properly
name_input = [0]*n_input
name_output = [0]*n_output
variation = float(workbook.sheets['Sheet1'].range(23, 13).value)  # Variable limits defined by user as a % of nominal.
limits = np.zeros([n_input, 5], dtype=float)  # Variable to store nominal value and limits of a variable
sampling_limits = np.zeros([n_input, 2], dtype=float)  # Stores final limits after

mean = np.zeros([n_input], dtype=float)
std_min = np.zeros([n_input], dtype=float)
std_max = np.zeros([n_input], dtype=float)
dif_min = np.zeros([n_input], dtype=float)
dif_max = np.zeros([n_input], dtype=float)
bound_low = np.zeros([n_input], dtype=float)
bound_high = np.zeros([n_input], dtype=float)
skew_factor_1: ndarray = np.zeros([n_input], dtype=float)
skew_factor_2: ndarray = np.zeros([n_input], dtype=float)
skew_factor_3: ndarray = np.zeros([n_input], dtype=float)

# Depending on data source code execution changes
if data_source == 1:  # Checks if data source is Hysys (data_source = 1)

    # Defining the name of the Input and Output variables and getting variable boundaries
    for n in range(n_input):
        name_input[n] = workbook.sheets['Sheet1'].range(n+4, 3).value + '.' + workbook.sheets['Sheet1'].range(n+4, 4)\
            .value
        limits[n, 0] = workbook.sheets['Sheet1'].range(n+4, 7).value  # Nominal value of variable
        limits[n, 3] = workbook.sheets['Sheet1'].range(n+4, 6).value  # Type of boundary (Tiresias_Book or User)
        limits[n, 4] = workbook.sheets['Sheet1'].range(n+4, 2).value  # Variable changing or not

        if limits[n, 0] == 0:  # Zeros defined as small value to avoid errors in program

            limits[n, 0] = 1E-10

        if limits[n, 4] == 2:  # If variable not changing, limits set to nominal.

            limits[n, 1] = limits[n, 0]  # Minimum value
            limits[n, 2] = limits[n, 0]  # Maximum value

        else:

            if limits[n, 3] == 2:  # If data comes from User, takes the minimum and maximum from excel

                limits[n, 1] = workbook.sheets['Sheet1'].range(n + 4, 8).value  # Minimum value
                limits[n, 2] = workbook.sheets['Sheet1'].range(n + 4, 9).value  # Maximum value

            else:  # If data comes from Tiresias_Book, defines limits based on predefined variation percentage

                limits[n, 1] = limits[n, 0] - abs(variation * limits[n, 0])  # Minimum value
                limits[n, 2] = limits[n, 0] + abs(variation * limits[n, 0])  # Maximum value

    for n in range(n_output):
        name_output[n] = workbook.sheets['Sheet1'].range(n+4, 17).value + '.' + workbook.sheets['Sheet1'].range(n+4, 18).value

    # A second run through the input matrix to delete non - changing variables (to help optimized sampling models)
    # It also removes user variable which limits are set equal (i.e. recycles and adjusters)
    n_input_aux = n_input
    limits_backup = limits
    n = 0
    name_input_2 = name_input

    while n < n_input_aux:
        # If variable not changing, or set equal by user, removes it from the sampling space
        if limits[n, 4] == 2 or (limits[n, 3] == 2 and limits[n, 1] == limits[n, 2]):
            limits = np.delete(limits, n, 0)
            name_input_2 = np.delete(name_input_2, n, 0)
            n -= 1
            n_input_aux -= 1

        n += 1

    name_input_2 = np.array([name_input_2])

    # IDEA FOR LATER: Variation must be modified if convergence window changes.
    # LATER ALSO: Suggest modifications of user-defined minimum or maximum if narrow convergence window.

    np.set_printoptions(suppress=True)
    print("Input variables:\n")
    print(tabulate(np.concatenate((name_input_2.T, np.round(np.array(limits[:, 0:3]), decimals=2)), axis=1),
                   headers=['Variable', 'Nominal', 'Minimum', 'Maximum']))
    print("\nDesign of Experiments (DoE) in progress...\n")
    sampling_limits = limits[:, [1, 2]]

    # Defining the skewness factor for Normally distributed methods
    for n in range(n_input_aux):
        mean[n] = limits[n, 0]
        dif_min[n] = abs(limits[n, 0] - limits[n, 1])   # Finds absolute difference with minimum limit
        dif_max[n] = abs(limits[n, 0] - limits[n, 2])   # Finds absolute difference with maximum limit
        std_min[n] = 0.341 * min(dif_min[n], dif_max[n])   # Smallest STD obtained via -Empirical rule-
        std_max[n] = 0.341 * max(dif_min[n], dif_max[n])   # Biggest STD obtained via -Empirical rule-

    for n in range(n_input_aux):

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

        # *Skew factor calculation made with its definition (Third moment around the mean)

    # Sampling using Scipy (To improve Latin Hypercube DoE). Generates basis for LHS and Normally distributed methods
    lhd = qmc.LatinHypercube(d=n_input_aux).random(n=n_simulations)  # Defines a matrix of coefficients for point dist.
    lhd2 = qmc.LatinHypercube(d=n_input_aux).random(n=n_simulations_2)

    # Opens Tiresias_Data and puts the name of the input and output variables
    workbook_2 = xw.Book('Tiresias_Data.xlsx')
    for n in range(n_input):
        workbook_2.sheets['Sheet1'].range(1, n+1).value = name_input[n]  # Write input variables names
    for n in range(n_output):
        workbook_2.sheets['Sheet1'].range(1, n+1+n_input).value = name_output[n]  # Write output variables names

    # Sampling depending on main chosen sampling method

    if sampling_method == 1:
        print("Sampling method: Latin Hypercube")
        sampling_LHS = LHS(xlimits=sampling_limits)
        x_LHS = sampling_LHS(n_simulations)
        label_1 = "Latin Hypercube"
        x_1 = x_LHS
        data_export = x_LHS
    elif sampling_method == 2:
        print("Sampling method: Random")
        sampling_RND = Random(xlimits=sampling_limits)
        x_RND = sampling_RND(n_simulations)
        label_1 = "Random"
        x_1 = x_RND
        data_export = x_RND
    elif sampling_method == 3:
        print("Sampling method: LHS by skopt")
        sampling_LHS_skopt = skopt.sampler.lhs.Lhs(lhs_type="classic", criterion=None)
        x_LHS_skopt = np.array(sampling_LHS_skopt.generate(dimensions=sampling_limits, n_samples=n_simulations))
        label_1 = "Latin Hypercube (skopt)"
        x_1 = x_LHS_skopt
        data_export = x_LHS_skopt
    elif sampling_method == 4:
        print("Sampling method: Normal Distribution")
        x_norm = skewnorm.ppf(lhd, 0, loc=mean, scale=std_max)  # Uses normal distribution
        label_1 = "Normal Distribution"
        x_1 = x_norm
        data_export = x_norm
    elif sampling_method == 5:
        print("Sampling method: Skewed normal X1")
        x_skew_1 = skewnorm.ppf(lhd, skew_factor_1, loc=mean, scale=std_max)
        label_1 = "Skewed normal X1"
        x_1 = x_skew_1
        data_export = x_skew_1
    elif sampling_method == 6:
        print("Sampling method: Skewed normal X2")
        x_skew_2 = skewnorm.ppf(lhd, skew_factor_2, loc=mean, scale=std_max)
        label_1 = "Skewed normal X2"
        x_1 = x_skew_2
        data_export = x_skew_2
    elif sampling_method == 7:
        print("Sampling method: Skewed normal X3")
        x_skew_3 = skewnorm.ppf(lhd, skew_factor_3, loc=mean, scale=std_max)
        label_1 = "Skewed normal X3"
        x_1 = x_skew_3
        data_export = x_skew_3
    elif sampling_method == 8:
        print("Sampling method: Sobol")
        sampling_Sobol = skopt.sampler.Sobol()
        x_Sobol = np.array(sampling_Sobol.generate(dimensions=sampling_limits, n_samples=n_simulations))
        label_1 = "Sobol"
        x_1 = x_Sobol
        data_export = x_Sobol
    elif sampling_method == 9:
        print("Sampling method: Centered LHS")
        sampling_LHS_center = skopt.sampler.lhs.Lhs(lhs_type="centered", criterion=None)
        x_LHS_center = np.array(sampling_LHS_center.generate(dimensions=sampling_limits, n_samples=n_simulations))
        label_1 = "Centered LHS"
        x_1 = x_LHS_center
        data_export = x_LHS_center
    elif sampling_method == 10:
        print("Sampling method: Maximin optimized LHS")
        sampling_LHS_maximin = skopt.sampler.lhs.Lhs(criterion="maximin", iterations=10000)
        x_LHS_maximin = np.array(sampling_LHS_maximin.generate(dimensions=sampling_limits, n_samples=n_simulations))
        label_1 = "Maximin optimized LHS"
        x_1 = x_LHS_maximin
        data_export = x_LHS_maximin
    elif sampling_method == 11:
        print("Sampling method: Correlation optimized LHS")
        sampling_LHS_correlation = skopt.sampler.lhs.Lhs(criterion="correlation", iterations=10000)
        x_LHS_correlation = np.array(sampling_LHS_correlation.generate(dimensions=sampling_limits, n_samples=n_simulations))
        label_1 = "Correlation optimized LHS"
        x_1 = x_LHS_correlation
        data_export = x_LHS_correlation
    elif sampling_method == 12:
        print("Sampling method: Ratio optimized LHS")
        sampling_LHS_ratio = skopt.sampler.lhs.Lhs(criterion="ratio", iterations=10000)
        x_LHS_ratio = np.array(sampling_LHS_ratio.generate(dimensions=sampling_limits, n_samples=n_simulations))
        label_1 = "Ratio optimized LHS"
        x_1 = x_LHS_ratio
        data_export = x_LHS_ratio
    elif sampling_method == 13:
        print("Sampling method: Halton")
        sampling_Halton = skopt.sampler.Halton()
        x_Halton = np.array(sampling_Halton.generate(dimensions=sampling_limits, n_samples=n_simulations))
        label_1 = "Halton"
        x_1 = x_Halton
        data_export = x_Halton
    elif sampling_method == 14:
        print("Sampling method: Hammersly")
        sampling_Hammersly = skopt.sampler.Hammersly()
        x_Hammersly = np.array(sampling_Hammersly.generate(dimensions=sampling_limits, n_samples=n_simulations))
        label_1 = "Hammersly"
        x_1 = x_Hammersly
        data_export = x_Hammersly
    elif sampling_method == 15:
        print("Sampling method: Grid")
        sampling_Grid = skopt.sampler.Grid(border="include", use_full_layout=False)
        x_Grid  = np.array(sampling_Grid .generate(dimensions=sampling_limits, n_samples=n_simulations))
        label_1 = "Grid"
        x_1 = x_Grid
        data_export = x_Grid

    print(n_simulations, "samples generated")

    # Sampling depending on secondary chosen sampling method (1 = None)

    if sampling_method_2 == 2:
        print("Secondary sampling method: Latin Hypercube")
        sampling_LHS = LHS(xlimits=sampling_limits)
        x_LHS = sampling_LHS(n_simulations_2)
        label_2 = "Latin Hypercube"
        x_2 = x_LHS
        print(type(x_2))
        data_export_2 = x_LHS
    elif sampling_method_2 == 3:
        print("Secondary sampling method: Random")
        sampling_RND = Random(xlimits=sampling_limits)
        x_RND = sampling_RND(n_simulations_2)
        label_2 = "Random"
        x_2 = x_RND
        data_export_2 = x_RND
    elif sampling_method_2 == 4:
        print("Sampling method: LHS by skopt")
        sampling_LHS_skopt = skopt.sampler.lhs.Lhs(lhs_type="classic", criterion=None)
        x_LHS_skopt = np.array(sampling_LHS_skopt.generate(dimensions=sampling_limits, n_samples=n_simulations_2))
        label_2 = "Latin Hypercube (skopt)"
        x_2 = x_LHS_skopt
        data_export_2 = x_LHS_skopt
    elif sampling_method_2 == 5:
        print("Secondary sampling method: Normal Distribution")
        x_norm = skewnorm.ppf(lhd2, 0, loc=mean, scale=std_max)  # Uses normal distribution
        label_2 = "Normal Distribution"
        x_2 = x_norm
        data_export_2 = x_norm
    elif sampling_method_2 == 6:
        print("Secondary sampling method: Skewed normal X1")
        x_skew_1 = skewnorm.ppf(lhd2, skew_factor_1, loc=mean, scale=std_max)
        label_2 = "Skewed normal X1"
        x_2 = x_skew_1
        data_export_2 = x_skew_1
    elif sampling_method_2 == 7:
        print("Secondary sampling method: Skewed normal X2")
        x_skew_2 = skewnorm.ppf(lhd2, skew_factor_2, loc=mean, scale=std_max)
        label_2 = "Skewed normal X2"
        x_2 = x_skew_2
        data_export_2 = x_skew_2
    elif sampling_method_2 == 8:
        print("Secondary sampling method: Skewed normal X3")
        x_skew_3 = skewnorm.ppf(lhd2, skew_factor_3, loc=mean, scale=std_max)
        label_2 = "Skewed normal X3"
        x_2 = x_skew_3
        data_export_2 = x_skew_3
    elif sampling_method_2 == 9:
        print("Sampling method: Sobol")
        sampling_Sobol = skopt.sampler.Sobol()
        x_Sobol = np.array(sampling_Sobol.generate(dimensions=sampling_limits, n_samples=n_simulations_2))
        label_2 = "Sobol"
        x_2 = x_Sobol
        data_export_2 = x_Sobol
    elif sampling_method_2 == 10:
        print("Sampling method: Centered LHS")
        sampling_LHS_center = skopt.sampler.lhs.Lhs(lhs_type="centered", criterion=None)
        x_LHS_center = np.array(sampling_LHS_center.generate(dimensions=sampling_limits, n_samples=n_simulations_2))
        label_2 = "Centered LHS"
        x_2 = x_LHS_center
        data_export_2 = x_LHS_center
    elif sampling_method_2 == 11:
        print("Sampling method: Maximin optimized LHS")
        sampling_LHS_maximin = skopt.sampler.lhs.Lhs(criterion="maximin", iterations=10000)
        x_LHS_maximin = np.array(sampling_LHS_maximin.generate(dimensions=sampling_limits, n_samples=n_simulations_2))
        label_2 = "Maximin optimized LHS"
        x_2 = x_LHS_maximin
        data_export_2 = x_LHS_maximin
    elif sampling_method_2 == 12:
        print("Sampling method: Correlation optimized LHS")
        sampling_LHS_correlation = skopt.sampler.lhs.Lhs(criterion="correlation", iterations=10000)
        x_LHS_correlation = np.array(sampling_LHS_correlation.generate(dimensions=sampling_limits, n_samples=n_simulations_2))
        label_2 = "Correlation optimized LHS"
        x_2 = x_LHS_correlation
        data_export_2 = x_LHS_correlation
    elif sampling_method_2 == 13:
        print("Sampling method: Ratio optimized LHS")
        sampling_LHS_ratio = skopt.sampler.lhs.Lhs(criterion="ratio", iterations=10000)
        x_LHS_ratio = np.array(sampling_LHS_ratio.generate(dimensions=sampling_limits, n_samples=n_simulations_2))
        label_2 = "Ratio optimized LHS"
        x_2 = x_LHS_ratio
        data_export_2 = x_LHS_ratio
    elif sampling_method_2 == 14:
        print("Sampling method: Halton")
        sampling_Halton = skopt.sampler.Halton()
        x_Halton = np.array(sampling_Halton.generate(dimensions=sampling_limits, n_samples=n_simulations_2))
        label_2 = "Halton"
        x_2 = x_Halton
        data_export_2 = x_Halton
    elif sampling_method_2 == 15:
        print("Sampling method: Hammersly")
        sampling_Hammersly = skopt.sampler.Hammersly()
        x_Hammersly = np.array(sampling_Hammersly.generate(dimensions=sampling_limits, n_samples=n_simulations_2))
        label_2 = "Hammersly"
        x_2 = x_Hammersly
        data_export_2 = x_Hammersly
    elif sampling_method_2 == 16:
        print("Sampling method: Grid")
        sampling_Grid = skopt.sampler.Grid(use_full_layout=False)
        x_Grid = np.array(sampling_Grid.generate(dimensions=sampling_limits, n_samples=n_simulations_2))
        label_2 = "Grid"
        x_2 = x_Grid
        data_export_2 = x_Grid

    # Data export to Tiresias_Data with 3 comma positions
    if sampling_method_2 == 1:
        data_export_complete = np.around(data_export, decimals=4)
    else:
        data_export_complete = np.around(np.vstack((data_export, data_export_2)), decimals=3)
        print(n_simulations_2, "samples generated")

    # Find the discrepancy to evaluate the space coverage
    # Una vez que se logra a un valor aceptable de la L2, nos detenemos cuando llegue al 90% de la metrica
    # Feature extraction non va, ya que se hizo feature reduction

    # Restore the non - changing variables with the nominal value to finally export data

    n_aux = 0
    Final_data = np.zeros([n_simulations + n_simulations_2, n_input], dtype=float)
    print("\nInitial variables: %i" % n_input)
    print("Black box model inputs (variables which change): %i" % n_input_aux)

    if sampling_method_2 == 1:
        Normalized_Data = qmc.scale(data_export_complete, limits[:, 1], limits[:, 2], reverse=True)
        print('\nMixture Discrepancy (MD): ',
              np.round(qmc.discrepancy(Normalized_Data, method='MD'), decimals=5))  # Mixture CD + WD
    else:
        Normalized_Data_1 = qmc.scale(data_export, limits[:, 1], limits[:, 2], reverse=True)
        Normalized_Data_2 = qmc.scale(data_export_2, limits[:, 1], limits[:, 2], reverse=True)
        print('\nMixture Discrepancy (MD)', label_1, ':', np.round(qmc.discrepancy(Normalized_Data_1, method='MD'), decimals=5))  # Mixture CD + WD
        print('Mixture Discrepancy (MD)', label_2, ':',
              np.round(qmc.discrepancy(Normalized_Data_2, method='MD'), decimals=5))  # Mixture CD + WD

    ''' # Other Discrepancy measurements
    print('CD:', qmc.discrepancy(Normalized_Data, method='CD'))  # Centered L2
    print('WD:', qmc.discrepancy(Normalized_Data, method='WD'))  # Wrap around discrepancy
    print('L2:', qmc.discrepancy(Normalized_Data, method='L2-star'))  # L2 discrepancy
    '''

    # print('\nL2 Discrepancy:', qmc.discrepancy(Normalized_Data, method='L2-star'))  # L2 discrepancy

    for n in range(n_input):

        if limits_backup[n, 4] == 2 or (limits_backup[n, 3] == 2 and limits_backup[n, 1] == limits_backup[n, 2]):

            Final_data[:, n] = limits_backup[n, 0]

        else:  # If value is changing, insert it in the matrix to export to excel, using an auxiliary counter

            Final_data[:, n] = data_export_complete[:, n_aux]
            n_aux += 1

    workbook_2.sheets['Sheet1'].range(2, 1).value = Final_data
    workbook_2.save()

else:  # For data source Real plant data (data_source = 2)

    n_files = np.int(workbook.sheets['Sheet1'].range(2, 13).value)  # Extraction of number of files from excel
    paths = [0] * n_files
    books = [0] * n_files
    data_export = [0] * (n_input + n_output)
    print("Data extraction in progress...")
    xw.App().visible = False

    for i in range(n_files):

        paths[i] = workbook.sheets['Sheet1'].range(i + 2, 11).value  # Paths written in specific rows in "Sheet1"

    book = xw.Book(paths[0])  # Opens the book with the Sensors information
    n_data = np.int(book.sheets[0].range('B2').current_region.end('down').row - 100)  #

    # Defining the name of the Input and Output variables and getting variable values
    for n in range(n_input):  # For every variable

        name_input[n] = workbook.sheets['Sheet1'].range(n+4, 3).value

        for i in range(n_files):  # For every workbook

            book = xw.Book(paths[i])

            for j in range(book.sheets.count):  # For every worksheet

                if book.sheets[j].name == name_input[n]:  # If name of the variable coincides

                    real_data = book.sheets[name_input[n]].range((2, 2), (n_data, 2)).value

        try:
            real_data
        except NameError:
            print('Data has not been found for variable', name_input[n])
            quit()

        data_export[n] = real_data

    print("Input variables read")

    for n in range(n_output):

        name_output[n] = workbook.sheets['Sheet1'].range(n+4, 17).value

        for i in range(n_files):  # For every workbook

            book = xw.Book(paths[i])

            for j in range(book.sheets.count):  # For every worksheet

                if book.sheets[j].name == name_output[n]:  # If name of the variable coincides

                    real_data = book.sheets[name_output[n]].range((2, 2), (n_data, 2)).value

            book.close()

        try:
            real_data
        except NameError:
            print('Data has not been found for variable', name_input[n])
            quit()

        data_export[n + n_input] = real_data

    print("Output variables read")

    workbook_2 = xw.Book('Tiresias_Data.xlsx')
    for n in range(n_input):
        workbook_2.sheets['Sheet1'].range(1, n+1).value = name_input[n]
    for n in range(n_output):
        workbook_2.sheets['Sheet1'].range(1, n+1+n_input).value = name_output[n]

    workbook_2.sheets['Sheet1'].range(2, 1).value = np.array(data_export).T
    workbook_2.save()

print("\n--- Execution finished after: %s seconds ---\n" % np.round(time.time() - start_time, decimals=2))

# Plotting zone - Plots the sampling points of the first two variables (2D plot)
plot_yn = input('Plot the results? (y/n): ')

if plot_yn == 'y':

    # LATER: 3D plot would be good also

    # Which variable will be plotted
    temp1, temp2 = limits.shape
    var1 = 0
    var2 = temp1 - 1

    font = {'family': 'Times New Roman',
            'size': 18}
    plt.rc('font', **font)
    plt.rcParams['figure.constrained_layout.use'] = True

    # plt.title("Design of experiments")
    plt.xlim([limits[var1, 1] - abs(0.1 * limits[var1, 0]), limits[var1, 2] + abs(0.1 * limits[var1, 0])])
    plt.ylim([limits[var2, 1] - abs(0.1 * limits[var2, 0]), limits[var2, 2] + abs(0.1 * limits[var2, 0])])
    plt.xlabel(name_input[var1])
    plt.ylabel(name_input[var2])
    plt.plot(x_1[:, var1], x_1[:, var2], "o", label=label_1)
    if sampling_method_2 != 1:
        plt.plot(x_2[:, var1], x_2[:, var2], "o", label=label_2)

    plt.axvline(x=limits[var1, 1], color='r', linestyle="--", linewidth=0.7)
    plt.axvline(x=limits[var1, 2], color='r', linestyle="--", linewidth=0.7)
    plt.axvline(x=limits[var1, 0], color='g', linestyle="--", linewidth=0.7)
    plt.axhline(y=limits[var2, 1], color='r', linestyle="--", linewidth=0.7)
    plt.axhline(y=limits[var2, 2], color='r', linestyle="--", linewidth=0.7)
    plt.axhline(y=limits[var2, 0], color='g', linestyle="--", linewidth=0.7)
    plt.legend(ncol=5)
    plt.show()

    '''
    plt.subplot(1, 3, 1)
    plt.xlim([limits[0, 1] - 1, limits[0, 2] + 1])
    plt.ylim([limits[1, 1] - 0.1, limits[1, 2] + 0.1])
    plt.title("LHS, RND and FF")
    plt.plot(x_LHS[:, 0], x_LHS[:, 1], "o", label="LHS")
    plt.plot(x_RND[:, 0], x_RND[:, 1], "o", label="Random")
    plt.plot(x_FF[:, 0], x_FF[:, 1], "o", label="Full Factorial")
    plt.axvline(x=limits[0, 1], color='r', linestyle="--", linewidth=0.7)
    plt.axvline(x=limits[0, 2], color='r', linestyle="--", linewidth=0.7)
    plt.axvline(x=limits[0, 0], color='g', linestyle="--", linewidth=0.7)
    plt.axhline(y=limits[1, 1], color='r', linestyle="--", linewidth=0.7)
    plt.axhline(y=limits[1, 2], color='r', linestyle="--", linewidth=0.7)
    plt.axhline(y=limits[1, 0], color='g', linestyle="--", linewidth=0.7)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    plt.subplot(1, 3, 2)
    plt.xlim([limits[0, 1] - 1, limits[0, 2] + 1])
    plt.ylim([limits[1, 1] - 0.1, limits[1, 2] + 0.1])
    plt.axvline(x=limits[0, 1], color='r', linestyle="--", linewidth=0.7)
    plt.axvline(x=limits[0, 2], color='r', linestyle="--", linewidth=0.7)
    plt.axvline(x=limits[0, 0], color='g', linestyle="--", linewidth=0.7)
    plt.axhline(y=limits[1, 1], color='r', linestyle="--", linewidth=0.7)
    plt.axhline(y=limits[1, 2], color='r', linestyle="--", linewidth=0.7)
    plt.axhline(y=limits[1, 0], color='g', linestyle="--", linewidth=0.7)
    plt.title("LH with Normal Distribution")
    plt.plot(sample2[:, 0], sample2[:, 1], "o", label="LHS")

    plt.subplot(1, 3, 3)
    plt.xlim([limits[0, 1] - 1, limits[0, 2] + 1])
    plt.ylim([limits[1, 1] - 0.1, limits[1, 2] + 0.1])
    plt.axvline(x=limits[0, 1], color='r', linestyle="--", linewidth=0.7)
    plt.axvline(x=limits[0, 2], color='r', linestyle="--", linewidth=0.7)
    plt.axvline(x=limits[0, 0], color='g', linestyle="--", linewidth=0.7)
    plt.axhline(y=limits[1, 1], color='r', linestyle="--", linewidth=0.7)
    plt.axhline(y=limits[1, 2], color='r', linestyle="--", linewidth=0.7)
    plt.axhline(y=limits[1, 0], color='g', linestyle="--", linewidth=0.7)
    plt.title("LH with Skewed Normal Distribution")
    plt.plot(sample[:, 0], sample[:, 1], "o", label="LHS")
    plt.show()
    '''

print("\nPress Enter to close...")
end = input()
