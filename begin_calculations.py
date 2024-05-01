
# import time
# def run_exe(num):

#     zeros_num = 7 - len(str(num))
#     num_code = str(0) * zeros_num + str(num)
#     subprocess.run(["pars\original_model.exe", num_code])
#
# if __name__ == "__main__":
#     start = time.time()
#     # Define the range of numbers
#     numbers_range = range(1, 2)  # Adjust this range as needed
#
#     # Use a ThreadPoolExecutor to run the function concurrently
#     with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
#         # Submit the function for each number in the range
#         results = executor.submit(run_exe, numbers_range,timeout=50,chunksize=1)
#
#
#     end = time.time()
#     print(f"Time to evaluate one set of reaction conditions {(end - start)/60} minutes")
#     print("All subprocesses completed.")


# import sys
# sys.path.append("workers\worker0")
# sys.path.append("workers\worker1")
# sys.path.append("workers\worker2")
# sys.path.append("workers\worker3")
# sys.path.append("workers\worker4")
# sys.path.append("workers\worker5")
# import start_worker
# import start_worker1
# import start_worker2
# import start_worker3
# import start_worker4
# import start_worker5
