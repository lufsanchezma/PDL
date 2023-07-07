import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import xlwings as xw
import os

Models = xw.Book('Tiresias_Models.xlsx')

font = {'family' : 'Times New Roman',
        'size'   : 18}
plt.rc('font', **font)
plt.rcParams['figure.constrained_layout.use'] = True

# PLOT 1: Discrepancy of LHS and Maximin LHS
x1_1 = Models.sheets["Normal plots"].range("G3", "G15").value
y1_1 = Models.sheets["Normal plots"].range("H3", "H15").value
y1_2 = Models.sheets["Normal plots"].range("J3", "J15").value
y1_3 = Models.sheets["Normal plots"].range("I3", "I15").value
y1_4 = Models.sheets["Normal plots"].range("K3", "K15").value

plt.plot(x1_1, y1_1, 'o', x1_1, y1_2, 'o')  # Samples v.s. discrepancy
plt.legend(['LHS', 'Maximin'])
plt.xlabel('Number of samples')
plt.ylabel('Mixture discrepancy')
plt.show()
plt.plot(x1_1, y1_3, 'o', x1_1, y1_4, 'o')  # Samples v.s. time
plt.legend(['LHS', 'Maximin'])
plt.xlabel('Number of samples')
plt.ylabel('Sampling time (s)')
plt.show()


# NORMAL ERROR PLOTS

font = {'family' : 'Times New Roman',
        'size'   : 18}
plt.rc('font', **font)

# Plot between -10 and 10 with .001 steps.
x_rmse = np.arange(0, 0.15, 0.0005)
x_mae = np.arange(0, 0.15, 0.0005)

mean_rmse = Models.sheets["Normal plots"].range("B3", "B5").value
std_rmse = Models.sheets["Normal plots"].range("C3", "C5").value
mean_mae = Models.sheets["Normal plots"].range("D3", "D5").value
std_mae = Models.sheets["Normal plots"].range("E3", "E5").value

plt.plot(x_mae, norm.pdf(x_mae, mean_mae[0], std_mae[0]), 'k--',
         x_mae, norm.pdf(x_mae, mean_mae[1], std_mae[1]), 'k',
         x_mae, norm.pdf(x_mae, mean_mae[2], std_mae[2]), 'k:')

plt.legend(['SVR', 'ANN', 'Kriging'])
plt.xlabel('Normalized error')
plt.ylabel('Probability density')

plt.show()


# Code for Article plots (maybe useful later)
'''
ANN_Tot = 10
n_experiments = 8  # 30, 60, 120, 240, 360, 480, 600 and 1200

for j in range(n_experiments):

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
    ax1.set_xlabel('Normalized error')
    ax1.set_ylabel('Probability density')
    ax2.set_xlabel('Normalized error')

    # Calculating mean and standard deviation for RMSE
    mean_rmse = Models.sheets["Normal plots"].range((21, 6 + j * 8), (30, 6 + j * 8)).value
    std_rmse = Models.sheets["Normal plots"].range((21, 7 + j * 8), (30, 7 + j * 8)).value

    mean_mae = Models.sheets["Normal plots"].range((21, 8 + j * 8), (30, 8 + j * 8)).value
    std_mae = Models.sheets["Normal plots"].range((21, 9 + j * 8), (30, 9 + j * 8)).value

    for n in range(ANN_Tot):
        ax1.plot(x_rmse, norm.pdf(x_rmse, mean_rmse[n], std_rmse[n]))
        ax2.plot(x_mae, norm.pdf(x_mae, mean_mae[n], std_mae[n]))

    ax1.set_title('RMSE')
    ax2.set_title('MAE')
    ax1.legend(['ANN0', 'ANN1', 'ANN2', 'ANN3', 'ANN4', 'ANN5', 'ANN6', 'ANN7', 'ANN8', 'ANN9'], frameon=False)
    ax2.legend(['ANN0', 'ANN1', 'ANN2', 'ANN3', 'ANN4', 'ANN5', 'ANN6', 'ANN7', 'ANN8', 'ANN9'], frameon=False)

    if j == 0:
        plt.savefig(os.getcwd() + '\\Results' + '\\30_Error.pdf')
    elif j == 1:
        plt.savefig(os.getcwd() + '\\Results' + '\\60_Error.pdf')
    elif j == 2:
        plt.savefig(os.getcwd() + '\\Results' + '\\120_Error.pdf')
    elif j == 3:
        plt.savefig(os.getcwd() + '\\Results' + '\\240_Error.pdf')
    elif j == 4:
        plt.savefig(os.getcwd() + '\\Results' + '\\360_Error.pdf')
    elif j == 5:
        plt.savefig(os.getcwd() + '\\Results' + '\\480_Error.pdf')
    elif j == 6:
        plt.savefig(os.getcwd() + '\\Results' + '\\600_Error.pdf')
    elif j == 7:
        plt.savefig(os.getcwd() + '\\Results' + '\\1200_Error.pdf')
'''
