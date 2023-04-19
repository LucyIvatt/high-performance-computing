import os
import sys

START = 200
STEP = 400
END_SIZE = 9000
# os.system('just -l')

if sys.argv[1] == "original":
    prefix = "just viking_run original "
    for x in range(START, END_SIZE+STEP, STEP):
        y = int(x / 4)
        folder_name = str(f"x_{x}_y_{y}_orig")
        cmd = prefix + folder_name + " original_benchmarks" + f" -x {x} -y {y}"
        os.system(cmd)

elif sys.argv[1] == "openmp" and sys.argv[2] == "cpus":
    prefix = "just viking_run_openmp "
    for cpus in range(5, 45, 5):
        folder_name = str(f"{cpus}_cpus_omp ")
        cmd = prefix + folder_name + str(cpus) + " openmp_cpu_experiment"
        os.system(cmd)

elif sys.argv[1] == "cuda":
    pass

elif sys.argv[1] == "mpi":
    pass

else:
    print("Incorrect target name, please try again.")


