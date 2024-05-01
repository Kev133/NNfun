from scipy.stats import qmc
import numpy as np
import time
import os
import shutil
def generate_latin_hypercube(num_of_samples):
    sampler = qmc.LatinHypercube(d=3,seed=1)
    sample = sampler.random(n=num_of_samples)
    # [Tdeg,catalyst concentration ,cocatalyst concentration]
    l_bounds = [155,1.5,40] #120, 0.5 40
    u_bounds = [160,2,120]  #160 2 120


    training_data = qmc.scale(sample,l_bounds,u_bounds)
    Tdeg_data = training_data[:,0]
    cat_conc_data = training_data[:,1]
    cocat_conc_data = training_data[:,2]
    return Tdeg_data,cat_conc_data,cocat_conc_data

def generate_pars_files(data):

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
        "num_particles_mc_simulation": 5000
    }

    length_data = len(data[0])
    for i in range(0,length_data):
        default_values.update({"temp":data[0][i]})
        default_values.update({"catalyst":data[1][i]})
        default_values.update({"co_catalyst":data[2][i]})
        i = i + 1
        zeros_num = 7-len(str(i))
        num_code = str(0)*zeros_num+str(i)
        path = "pars\\par_"+num_code+".dat"
        file = open(path,"w")
        with open(path,'w') as f:
            for value in default_values.values():
                 f.write('{}\n'.format(value))
def distribute_pars_files():
    worker_num = 6
    time.sleep(2)
    files = os.listdir("pars")
    par_files = [file for file in files if file.startswith("par_")]
    pars_divided = []
    files_per_worker = int(len(par_files)/worker_num)
    i=0
    for j in range(0, worker_num):
        chunk = par_files[i:i + files_per_worker]

        pars_divided.append(chunk)
        i=i+files_per_worker

    for i in range(0,worker_num):
        for j in range(0,files_per_worker):
            shutil.copy("pars/" +pars_divided[i][j], "workers/worker"+str(i))



def delete_pars_workers():
    worker_dirs = os.listdir("workers")
    worker_list = [file for file in worker_dirs if file.startswith("worker")]
    for worker in worker_list:
        path = "workers/"+str(worker)
        worker_dirs = os.listdir(path)
        [os.remove(path+"/"+file) for file in worker_dirs if file.startswith("par")]
def delete_out_workers():
    worker_dirs = os.listdir("workers")
    worker_list = [file for file in worker_dirs if file.startswith("worker")]
    for worker in worker_list:
        path = "workers/"+str(worker)
        worker_dirs = os.listdir(path)
        [os.remove(path+"/"+file) for file in worker_dirs if file.startswith("out")]


num_of_samples = 19200
print(num_of_samples/6)
if num_of_samples%6==0:
    #lhs_nums = generate_latin_hypercube(num_of_samples)
    #generate_pars_files(lhs_nums)
    distribute_pars_files()
#delete_pars_workers()
#delete_out_workers()