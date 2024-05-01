import subprocess
import os
"""
WORKER 2
"""
pars_file_nums =None
files = os.listdir()
len_out_files = len([file for file in files if file.startswith("out")])
len_par_files = len([file for file in files if file.startswith("par")])

if len_out_files == 0:
    pars_file_nums = [int(file[4:11]) for file in files if file.startswith("par")]
elif len_out_files == len_par_files:
    print("every input parameter has been processed")


elif len_out_files < len_par_files:
    out_file_nums = [int(file[4:11]) for file in files if file.startswith("out")]
    pars_file_nums = [int(file[4:11]) for file in files if file.startswith("par")]

    difference = set(pars_file_nums).difference(out_file_nums) #finds the different values and puts them in a list
    difference.add(max(out_file_nums)) #adding the last value of out_files in case it was interupted in the last session
    print(difference)
    pars_file_nums=difference
else:
    print("PROGRAM NEMUZE SROVNAT KOLIK JE PAR A OUT SOUBORŮ")



def run_exe(num):
    zeros_num = 7 - len(str(num))
    num_code = str(0) * zeros_num + str(num)
    subprocess.run(["original_model.exe", num_code])
if pars_file_nums:
    for number in range(min(pars_file_nums),max(pars_file_nums)+1):
        run_exe(number)
        print(number)
    print("Worker2 done")