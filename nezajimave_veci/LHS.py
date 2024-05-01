from scipy.stats import qmc
from matplotlib import pyplot as plt
sampler = qmc.LatinHypercube(d=2,seed=14) #optimization= "lloyd"
sample = sampler.random(n=5)
# [Tdeg,catalyst concentration ,cocatalyst concentration]
l_bounds = [0,0]
u_bounds = [1,1] #above 170 is unstable

x = [0,0.2,0.4,0.6,0.8,1]
training_data = qmc.scale(sample,l_bounds,u_bounds)
Tdeg_data = training_data[:,0]
cat_conc_data = training_data[:,1]

plt.scatter(training_data[:,0],training_data[:,1],s=70)
plt.xticks(x,fontsize=15)
plt.yticks(x,fontsize=15)
plt.grid()
plt.show()