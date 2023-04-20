import os
import sys
import subprocess
import re

START = 100
STEP = 100
END_SIZE = 2000

if sys.argv[1] == "original":
    if input("Are you sure you want to run all original benchmarks on Viking? ") != "yes":
        exit()

    prefix = "just viking_run original "
    for x in range(START, END_SIZE+STEP, STEP):
        y = int(x / 4)
        folder_name = str(f"x_{x}_y_{y}_orig")
        cmd = prefix + folder_name + " original_benchmarks" + f" -x {x} -y {y}"

        if sys.argv[2] == "print":
            print(cmd)
        else:
            os.system(cmd)

elif sys.argv[1] == "openmp" and sys.argv[2] == "cpus":
    if input("Are you sure you want to run openmp cpu experiment on Viking? ") != "yes":
        exit()

    prefix = "just viking_run_openmp "
    for cpus in range(2, 42, 2):
        folder_name = str(f"{cpus}_cpus_omp ")
        cmd = prefix + folder_name + str(cpus) + " openmp_cpu_experiment"
        
        if sys.argv[3] == "print":
            print(cmd)
        else:
            os.system(cmd)
    
elif sys.argv[1] == "openmp" and sys.argv[2] == "benchmarks":
    if input("Are you sure you want to run openmp benchmarks on Viking? ") != "yes":
        exit()

    prefix = "just viking_run_openmp "
    for x in range(START, END_SIZE+STEP, STEP):
        y = int(x / 4)
        folder_name = str(f"x_{x}_y_{y}_omp")
        cmd = prefix + folder_name + " 20 openmp_benchmarks" + f" -x {x} -y {y}"
        if sys.argv[3] == "print":
            print(cmd)
        else:
            os.system(cmd)

elif sys.argv[1] == "cuda" and sys.argv[2] == "checkpoints":
    if input("Are you sure you want to run cuda checkpoint experiment on Viking? ") != "yes":
        exit()

    prefix = "just viking_run_cuda "
    for x in range(START, END_SIZE+STEP, STEP):
        y = int(x / 4)
        folder_name = str(f"x_{x}_y_{y}_cuda_cp")
        cmd = prefix + folder_name + " cuda_checkpoint_experiment" + f" -x {x} -y {y} -c"

        if sys.argv[3] == "print":
            print(cmd)
        else:
            os.system(cmd)


elif sys.argv[1] == "cuda" and sys.argv[2] == "benchmarks":
    if input("Are you sure you want to run cuda benchmarks on Viking? ") != "yes":
        exit()

    prefix = "just viking_run_cuda "
    for x in range(START, END_SIZE+STEP, STEP):
        y = int(x / 4)
        folder_name = str(f"x_{x}_y_{y}_cuda")
        cmd = prefix + folder_name + " cuda_benchmarks" + f" -x {x} -y {y}"
        
        if sys.argv[3] == "print":
            print(cmd)
        else:
            os.system(cmd)

elif sys.argv[1] == "mpi":
    pass

elif sys.argv[1] == "slurm_copy" and sys.argv[2] in ["original_benchmarks", "openmp_benchmarks", "openmp_cpu_experiment", "cuda_checkpoint_experiment", "cuda_benchmarks"]:
    path = "/mnt/c/Users/lucea/Documents/Github Repos/HIPC-Assessment/validation/" + sys.argv[2]
    csv_path = "/mnt/c/Users/lucea/Documents/Github Repos/HIPC-Assessment/validation/"+ sys.argv[2] + "_data.csv"

    if not os.path.exists(path):
        os.makedirs(path)
    
    if not os.path.isfile(csv_path):
        with open(csv_path, "x") as f:
            for line in sys.stdin:
                line = line.rstrip("\n")
                split_path = line.split("/")
                os.system(f"just viking_rsync_from '~/scratch/{sys.argv[2]}/{split_path[-2]}/{split_path[-1]}' '{path}/{split_path[-2]}.out'")

                with open(f"{path}/{split_path[-2]}.out", "r") as slurm_log:
                    for log_line in slurm_log:
                        match = re.search(r'Total Time:\s+(\d+\.\d+)', log_line)
                        if match:
                            ver = split_path[-2].split("_")

                            if sys.argv[2] == "openmp_cpu_experiment":
                                f.write(f"cpus={ver[0]}, {match.group(1)}\n")
                            else:
                                f.write(f"x={ver[1]} y={ver[3]}, {match.group(1)}\n")
                            break
                    else:
                        if sys.argv[2] == "openmp_cpu_experiment":
                                f.write(f"cpus={ver[0]}, {600}\n")
                        else:
                            f.write(f"x={ver[1]} y={ver[3]}, {600}\n")

    else:
        print("data.csv already exists, clear directory and retry.")
else:
    print("Incorrect script arguments, please try again.")


