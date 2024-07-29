from scipy.stats import qmc
from matplotlib import pyplot as plt
sampler = qmc.LatinHypercube(d=3,seed=14) #optimization= "lloyd"
sample = sampler.random(n=200)
#[Tdeg,catalyst concentration ,cocatalyst concentration]
l_bounds = [0,0,20]
u_bounds = [1,1,80] #above 170 is unstable

x = [0,0.2,0.4,0.6,0.8,1]
training_data = qmc.scale(sample,l_bounds,u_bounds)
Tdeg_data = training_data[:,0]
cat_conc_data = training_data[:,1]
cocat_conc_data = training_data[:,2]

plt.scatter(Tdeg_data,cat_conc_data, cocat_conc_data)
# plt.xticks(x,fontsize=15)
# plt.yticks(x,fontsize=15)
# plt.grid()
plt.show()