import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import xlwings as xw
import os

Data = xw.Book('Tiresias_Data_S.A..xlsx')

font = {'family' : 'Times New Roman',
        'size'   : 18}
plt.rc('font', **font)
plt.rcParams['figure.constrained_layout.use'] = True

x1 = Data.sheets["Sheet1"].range("A9", "A49").value
y1 = Data.sheets["Sheet1"].range("AL9", "AL49").value

x2 = Data.sheets["Sheet1"].range("B50", "B90").value
y2 = Data.sheets["Sheet1"].range("AL50", "AL90").value

x3 = Data.sheets["Sheet1"].range("C91", "C139").value
y3 = Data.sheets["Sheet1"].range("AL91", "AL139").value

x4 = Data.sheets["Sheet1"].range("D140", "D220").value
y4 = Data.sheets["Sheet1"].range("AL140", "AL220").value

x5 = Data.sheets["Sheet1"].range("E221", "E261").value
y5 = Data.sheets["Sheet1"].range("AL221", "AL261").value

x6 = Data.sheets["Sheet1"].range("F262", "F278").value
y6 = Data.sheets["Sheet1"].range("AL262", "AL278").value

x7 = Data.sheets["Sheet1"].range("G279", "G295").value
y7 = Data.sheets["Sheet1"].range("AL279", "AL295").value

# plt.legend(['SVR', 'ANN', 'Kriging'])
plt.plot(x1, y1, 'k--', x2, y2, 'k', x3, y3, 'k:', x6, y6)  # Samples v.s. discrepancy
plt.show()
plt.plot(x4, y4, 'k*', x5, y5, 'ko', x7, y7, 'k^')  # Samples v.s. discrepancy
plt.legend(['Ethane', 'Propane', 'n-Butane'])
plt.xlabel('Molar fraction (%mol)')
plt.ylabel('Energy consumption (W)')
plt.show()


# 1 AND 2 NO CORRELATION WITH ENERGY CONSUMPTION
# 3 AND 6 NOT DEPENDENT
# 4 INCREASES
# 5 DECREASES
# 7 INCREASES
