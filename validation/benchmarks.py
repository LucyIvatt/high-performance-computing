import os
import sys
import platform
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
        cmd = prefix + folder_name + " 40 openmp_benchmarks" + f" -x {x} -y {y}"
        if sys.argv[3] == "print":
            print(cmd)
        else:
            os.system(cmd)

elif sys.argv[1] == "cuda" and sys.argv[2] == "checkpoints":
    if input("Are you sure you want to run cuda checkpoint experiment on Viking? ") != "yes":
        exit()

    if(sys.argv[3] == "default"):
        prefix = "just viking_run_cuda "
        for x in range(START, END_SIZE+STEP, STEP):
            y = int(x / 4)
            folder_name = str(f"x_{x}_y_{y}_cuda_cp")
            cmd = prefix + folder_name + " cuda_checkpoint_experiment" + f" -x {x} -y {y} -c"

            if sys.argv[4] == "print":
                print(cmd)
            else:
                os.system(cmd)
    elif sys.argv[3] == "extra":
        prefix = "just viking_run_cuda "
        for x in range(4000, 16000+2000, 2000):
            y = int(x / 4)
            folder_name = str(f"x_{x}_y_{y}_cuda_cp")
            cmd = prefix + folder_name + " cuda_checkpoint_experiment" + f" -x {x} -y {y} -c"

            if sys.argv[4] == "print":
                print(cmd)
            else:
                os.system(cmd)



elif sys.argv[1] == "cuda" and sys.argv[2] == "benchmarks":
    if input("Are you sure you want to run cuda benchmarks on Viking? ") != "yes":
        exit()

    if(sys.argv[3] == "default"):
        prefix = "just viking_run_cuda "
        for x in range(START, END_SIZE+STEP, STEP):
            y = int(x / 4)
            folder_name = str(f"x_{x}_y_{y}_cuda")
            cmd = prefix + folder_name + " cuda_benchmarks" + f" -x {x} -y {y}"
            
            if sys.argv[4] == "print":
                print(cmd)
            else:
                os.system(cmd)
    elif(sys.argv[3] == "extra"):
        prefix = "just viking_run_cuda "
        for x in range(2000, 16000+2000, 2000):
            y = int(x / 4)
            folder_name = str(f"x_{x}_y_{y}_cuda")
            cmd = prefix + folder_name + " cuda_benchmarks" + f" -x {x} -y {y}"
            
            if sys.argv[4] == "print":
                print(cmd)
            else:
                os.system(cmd)

elif sys.argv[1] == "mpi":
    pass

elif sys.argv[1] == "all" and sys.argv[2] == "default":
    commands = []
    for i in range(3):
        commands.append(f"just viking_run original original_run_{i} default_benchmarks")
        commands.append(f"just viking_run_cuda cuda_run_{i} default_benchmarks")
        commands.append(f"just viking_run_openmp openmp_run_{i} 20 default_benchmarks")

    for cmd in commands:   
        if sys.argv[3] == "print":
            print(cmd)
        else:
            os.system(cmd)



elif sys.argv[1] == "slurm_copy" and sys.argv[2] in ["original_benchmarks", "openmp_benchmarks", "openmp_cpu_experiment", "cuda_checkpoint_experiment", "cuda_benchmarks", "default_benchmarks"]:
    if platform.node() == "LUCE-PC":
        path = "/mnt/d/Libraries/Documents/Github Repos/HIPC-Assessment/validation/" + sys.argv[2]
    else:
        path = "/mnt/c/Users/lucea/Documents/Github Repos/HIPC-Assessment/validation/" + sys.argv[2]
    
    csv_path = path + "_data.csv"

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
                            elif sys.argv[2] == "default_benchmarks":
                                f.write(f"{split_path[-2]}, {match.group(1)}\n")

                            else:
                                f.write(f"x={ver[1]} y={ver[3]}, {match.group(1)}\n")
                            break
                    else:
                        if sys.argv[2] == "openmp_cpu_experiment":
                                f.write(f"cpus={ver[0]}, {600}\n")
                        elif sys.argv[2] == "default_benchmarks":
                                f.write(f"{split_path[-2]}, {600}\n")
                        else:
                            f.write(f"x={ver[1]} y={ver[3]}, {600}\n")

    else:
        data_pairs = {}
        for line in sys.stdin:
            line = line.rstrip("\n")
            split_path = line.split("/")
            os.system(f"just viking_rsync_from '~/scratch/{sys.argv[2]}/{split_path[-2]}/{split_path[-1]}' '{path}/{split_path[-2]}.out'")

            with open(f"{path}/{split_path[-2]}.out", "r") as slurm_log:
                    ver = split_path[-2].split("_")

                    for log_line in slurm_log:
                        match = re.search(r'Total Time:\s+(\d+\.\d+)', log_line)
                        if match:

                            if sys.argv[2] == "openmp_cpu_experiment":
                                data_pairs[f"cpus={ver[0]}"] = f"{match.group(1)}"
                            else:
                                data_pairs[f"x={ver[1]} y={ver[3]}"] = f"{match.group(1)}"
                            break
                    else:
                        if sys.argv[2] == "openmp_cpu_experiment":
                                data_pairs[f"cpus={ver[0]}"] = f"{600}"
                        else:
                            data_pairs[f"x={ver[1]} y={ver[3]}"] = f"{600}"
        with open(csv_path, "r+") as f:
            lines = f.readlines()
            new_lines = []
            for line in lines:
                if line.split(",")[0] in data_pairs.keys():
                    new_lines.append(line.strip("\n") + ", " + str(data_pairs[line.split(",")[0]]) + "\n")
                else:
                    new_lines.append(line)

            for key in data_pairs.keys():
                for line in lines:
                    if key in line:
                        break
                else:
                    new_lines.append(f"{key}, {data_pairs[key]}\n")
            f.seek(0)
            f.writelines(new_lines)
        
else:
    print("Incorrect script arguments, please try again.")


