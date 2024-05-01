
from matplotlib import pyplot as plt
import os
import statistics
import matplotlib.pyplot as plt
import numpy as np
Mn_ODE = []
Mn_MC = []
Mw_ODE = []
Mw_MC = []
G_crossover = []
G_plateau = []
list=[]
files = os.listdir()
# x = np.arange(0.1, 4, 0.5)
# y = np.exp(-x)
#
# fig, ax = plt.subplots()
# ax.errorbar(x, y, xerr=0.2, yerr=0.4)
# plt.show()
"""
Ehm, docela vnoreny hnus, v budoucnu nahradit funkcema
"""

for file in files:
    if "out" in file:
        print(file)
        with open(file, "r") as f:
            for line in f:
                if "(" in line:  # finds the correct lines
                    list.append(line)

            Mn_ODE.append(float(list[0][20:41]))
            Mn_MC.append(float(list[1][20:41]))
            Mw_ODE.append(float(list[2][20:41]))
            Mw_MC.append(float(list[3][20:41]))
            G_crossover.append(float(list[5][20:41]))
            G_plateau.append(float(list[5][20:41]))
            list=[]

MC_stovky=[]
MC_tisicovky=[]

for num in range (0,len(G_crossover)+1):
    if num < 1300:

        if num%100 == 0 and num!=0:
            MC_stovky.append(G_crossover[num-100:num])
    elif num < 1901:
        if num % 100 == 0 and num != 0:
            print(num)
            MC_tisicovky.append(G_crossover[num - 100:num])
    elif num < 231:
        pass
    else:
        print("Neco se posralo")
print(MC_stovky)
odchylky_stovky=[]
prumer_stovky=[]
variacni_koeficient=[]
odchylky_tisicovky=[]
prumer_tisicovky=[]
variacni_koeficient_tisicovky=[]

for stovka in MC_stovky:
    odchylky_stovky.append(statistics.stdev(stovka))
    prumer_stovky.append(statistics.mean(stovka))
    variacni_koeficient.append(statistics.stdev(stovka) / statistics.mean(stovka))
print("Stovky od 200 do 900\n")
print(f"Odchylky stovky {odchylky_stovky}")
print(f"prumer stovky{prumer_stovky}")
print(f"variacni koeficient stovky {variacni_koeficient}")
print("\nNásledují tísícovky od 1000 do 10 000\n")


# for tisicovka in MC_tisicovky:
#     odchylky_tisicovky.append(statistics.stdev(tisicovka))
#     prumer_tisicovky.append(statistics.mean(tisicovka))
#     variacni_koeficient_tisicovky.append(statistics.stdev(tisicovka) / statistics.mean(tisicovka))
# print(f"Odchylky tisicovky {odchylky_tisicovky}")
# print(f"prumer tisicovky{prumer_tisicovky}")
# print(f"variacni koeficient tisicovky {variacni_koeficient_tisicovky}")
#
# cisla=[200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,12000]
# odchylky = variacni_koeficient+variacni_koeficient_tisicovky
#
# plt.plot(cisla,odchylky,"-",markersize=0.8,label="huuh")
# plt.title("plateau modulus deviation profile ")
# plt.xlabel("number of simulated particles",fontsize = 10)
# plt.ylabel("plateau modulus standard deviation/mean",fontsize = 10)
# plt.show()