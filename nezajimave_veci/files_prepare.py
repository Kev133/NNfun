import numpy as np
"""
This module creates input data for the executable PLA program, the input data differs in the amount of particles? chains?
that are simulated. The user can choose how many files he wants to generate and how much should they vary.
"""

default_values= {
    "temp": 160.0,
    "catalyst": 1.0,
    "co_catalyst": 60,
    "octanoic_acid": 0.36,
    "activation_rate": 1.0e6,
    "pre_exp_propagation": 7.4e11,
    "activation_energy_propagation": 63.3,
    "heat_polymerization": -23.3,
    "entropy_polymerization": -22.0,
    "chain_transfer_rate": 1.0e6,
    "pre_exp_transesterification": 3.38e11,
    "activation_energy_transesterification": 83.3,
    "pre_exp_random_chain_scission": 1.69e8,
    "activation_energy_random_chain_scission": 101.5,
    "polymer_melt_temp": 200,
    "polymer_melt_density": 1.09,
    "chain_relaxation_time": 9.0e-7,
    "avg_monomer_units_entanglements": 65,
    "num_particles_mc_simulation": 7000,
    "num_chain_in_bob_simulation": 300
}

starting_point = 100
ending_point = 1000
number_of_files_for_one_MC_num = 100
hundreds = np.arange(starting_point,1200,100) # start, end, step
# thousands = np.arange(1000,10000,1000)
# ten_thousands = np.arange(10000,ending_point+2000,2000)

#list_raw = np.concatenate((hundreds,thousands,ten_thousands),axis=None)
list = []
for number in hundreds:
    for i in range(1,number_of_files_for_one_MC_num+1):
        list.append(number)

print(list)

i=1
for number in list:
    default_values.update({"num_chain_in_bob_simulation":number})
    zeros_num = 7-len(str(i))
    num_code = str(0)*zeros_num+str(i)
    path = "par_"+num_code+".dat"
    file = open(path,"w")
    i=i+1
    with open(path,'w') as f:
        for value in default_values.values():
            f.write('{}\n'.format(value))

