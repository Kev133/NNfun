import subprocess

def run_exe(num):

    zeros_num = 7 - len(str(num))
    num_code = str(0) * zeros_num + str(num)
    subprocess.run(["a.exe", num_code])
for num in range(1,10):
    print(num)
    run_exe(num)