#!/bin/env -S just --justfile
set positional-arguments
set dotenv-load

TARGETS := "original openmp cuda mpi"
ORIGINAL_TARGET := "original"
VORTEX_CMD := "./vortex"

# Viking-related folders and files
VIKING_TEMPLATE := "./viking_run.sh.j2"
VIKING_BENCH_DIR := "~/scratch/hipc_benches"
VIKING_BENCH_RESULTS_DIR := `mktemp -p /tmp -du hipcbenchXXX`

# Default viking run configuration
VIKING_MODULE := "compiler/GCC/11.2.0"
VIKING_PARTITION := "teach"
VIKING_SLURM_ARGS := ""
VIKING_JOB_TIME := "00:10:00"
VIKING_MEMORY := "4gb"
VIKING_NUM_TASKS := "1"
VIKING_CPUS_PT := "1"

LAB_MACHINE_ADDR := env_var("LAB_MACHINE_ADDR")
LAB_DEST := "hipc_source"

YORK_USER := env_var("YORK_USER")
export SSHPASS := env_var("YORK_PASS")

export OMP_NUM_THREADS := "5"

# Build a target
build target *make_args="": (clean target)
    (cd {{ target }} && make {{ make_args }})

# Run a target
run target *args="": (build target)
    "{{ target }}/vortex" {{ args }}

# Runs mpi
run_mpi num_p *args="": (build "mpi")
    mpirun -n {{ num_p }} ./mpi/vortex {{ args }}

# Benchmark a target against ORIGINAL_TARGET with `hyperfine`, no file I/O
hyperfine target *args="": (build target) (build ORIGINAL_TARGET)
    hyperfine --warmup 5 \
        "{{ target }}/vortex {{ args }} -n" \
        "{{ ORIGINAL_TARGET }}/vortex {{ args }} -n"

# Generate a flamegraph on a target, no file I/O
flame target *args="": (build target)
    flamegraph \
        -o "{{ target }}_flame.svg" \
        --deterministic \
        -- "{{ target }}/vortex" {{ args }} -n

# Use `perf` to profile a target, no file I/O
perf target *args="": (build target "CFLAGS=-g")
    rm -fv "{{ target }}_perf.data"
    perf record -g -s --output "{{ target }}_perf.data" -- \
        "{{ target }}/vortex" {{ args }} -n
    perf mem record -g -s --output "{{ target }}_mem_perf.data" -- \
        "{{ target }}/vortex" {{ args }} -n

# Profile a target with `scalasca`, no file I/O (won't work with cuda)
scalasca target *args="": (clean target) (build target "CC='scorep gcc'")
    (cd "{{ target }}" && scalasca -analyze {{ VORTEX_CMD }} {{ args }} -n)
    scalasca -examine "{{ target }}/scorep_vortex_OxO_sum"

# Run validator script on the output of two different targets
validate ltarget rtarget *args="": (build ltarget) (build rtarget)
    "{{ ltarget }}/vortex" {{ args }} \
        -o "{{ ltarget }}"
    "{{ rtarget }}/vortex" {{ args }} \
        -o "{{ rtarget }}"
    python3 ./validation/validator.py "{{ ltarget }}.vtk" "{{ rtarget }}.vtk"

# Run validator script on the output of mpi and another target
validate_mpi target num_p *args="": (build target) (build "mpi")
    "{{ target }}/vortex" {{ args }} \
        -o "{{ target }}"
    mpirun -n {{ num_p }} ./mpi/vortex {{ args }} \
        -o "mpi"
    python3 ./validation/validator.py "{{ target }}.vtk" "mpi.vtk"

# Clean build and run artefacts locally
clean *targets=TARGETS:
    -for target in {{ targets }}; \
        do (cd $target && make clean); \
    done
    find -iname "*.csv" -exec rm -v {} \;
    find -iname "*.vtk" -and -not -ipath "*/validation/*" -exec rm -v {} \;
    find -iname "*perf.data*" -exec rm -v {} \;
    find -iname "*.svg" -exec rm -v {} \;
    find -iname "*diffs.txt" -exec rm -rv {} \;
    -for target in {{ replace(targets, "viking", "") }}; do \
        ( cd $target && \
        find -iname "scorep*" -exec rm -rfv {} \; && \
        find -iname "vortex*input*" -exec rm -rfv {} \; ); \
    done
    find -maxdepth 1 -iname "scorep*" -exec rm -rfv {} \;
alias c := clean

# Call `rsync` on a lab machine
[private]
lab_rsync src dest *args="-r":
    sshpass -e rsync {{ args }} "{{ src }}" "{{ YORK_USER }}@{{ LAB_MACHINE_ADDR }}:{{ dest }}"

# Call `ssh` for a lab machine
lab_ssh *cmd="":
    sshpass -e ssh "{{ YORK_USER }}@{{ LAB_MACHINE_ADDR }}" '{{ cmd }}'

# Upload a target to a lab machine
lab_upload target: (clean target) (lab_rsync target LAB_DEST "-rI")

# Run a target on a lab machine
lab_run target *args="": (lab_upload target)
    just lab_ssh \
        'cd "{{ join(LAB_DEST, target) }}" \
        && make clean \
        && make \
        && {{ VORTEX_CMD }} {{ args }}'

# Run an arbitrary `just` target on a lab machine
lab_script *args="":
    #!/bin/bash
    set -euxo pipefail

    SCRIPT=$(mktemp /tmp/hipc_XXXXXX.sh)
    chmod +x $SCRIPT

    echo "cd {{ LAB_DEST }}" >> $SCRIPT
    just --dry-run --no-highlight {{ args }} 2>> $SCRIPT
    cat $SCRIPT
    just lab_rsync $SCRIPT $SCRIPT
    just lab_ssh "bash -c $SCRIPT"

# Upload a file to viking, defaults to recursive
[private]
viking_rsync_to src dest args="-r":
    sshpass -e rsync {{ args }} "{{ src }}" "{{ YORK_USER }}@viking.york.ac.uk:{{ dest }}"

# Download a file from viking
[private]
viking_rsync_from src dest args="":
    sshpass -e rsync {{ args }} "{{ YORK_USER }}@viking.york.ac.uk:{{ src }}" "{{ dest }}"

# Call `ssh` for viking
viking_ssh cmd="":
    sshpass -e ssh "{{ YORK_USER }}@viking.york.ac.uk" '{{ cmd }}'
alias vs := viking_ssh

# Run a target as a batch job on viking
viking_run target folder=`mktemp -du /tmp/hipcassessXXX` dest_folder="manual_tests" *args="": (clean target)
    mkdir -p "{{ folder }}"
    cd "{{ folder }}" && rm -rf "*" ".*"
    cp -rv "{{ target }}" "{{ folder }}"
    jinja2 \
        -o "{{ folder }}/run_{{ target }}.job" "{{ VIKING_TEMPLATE }}" \
        -D 'ntasks={{ VIKING_NUM_TASKS }}' \
        -D 'module={{ VIKING_MODULE }}' \
        -D 'partition={{ VIKING_PARTITION }}' \
        -D 'time_allot={{ VIKING_JOB_TIME }}' \
        -D 'cpus_pt={{ VIKING_CPUS_PT }}' \
        -D 'extra_opts={{ VIKING_SLURM_ARGS }}' \
        -D 'mem={{ VIKING_MEMORY }}' \
        -D build_cmd='(cd {{ target }} && make)' \
        -D run_cmd='(cd {{ target }} && {{ VORTEX_CMD }} {{ args }})'
    cat "{{ folder }}/run_{{ target }}.job"
    chmod +x "{{ folder }}/run_{{ target }}.job"
    just viking_rsync_to "{{ folder }}" "scratch/{{ dest_folder }}"
    just viking_ssh \
        'cd ~/scratch/{{ dest_folder }}/$(basename {{ folder }}) && \
        sbatch ./run_{{ target }}.job'
    @printf "\n==================================================\nViking job run in directory $(basename {{ folder }})\n\n"
    rm "{{ folder }}" -r

# Helper for viking_run for openmp
viking_run_openmp folder cpus="20" dest_folder="manual_tests" *args="":
    just \
        VIKING_JOB_TIME={{ VIKING_JOB_TIME }} \
        VIKING_CPUS_PT={{ cpus }} \
        VORTEX_CMD="OMP_NUM_THREADS={{ cpus }} {{ VORTEX_CMD }}" \
        viking_run "openmp" "{{ folder }}" {{dest_folder}} {{ args }}

# Helper for viking_run for cuda
viking_run_cuda folder dest_folder="manual_tests" *args="":
    just \
        VIKING_PARTITION=gpu \
        VIKING_SLURM_ARGS='#SBATCH --gres=gpu:1' \
        VIKING_JOB_TIME={{ VIKING_JOB_TIME }} \
        VIKING_MODULE=system/CUDA/11.1.1-GCC-10.2.0 \
        VIKING_CPUS_PT=1 \
        viking_run "cuda" "{{ folder }}" {{dest_folder}} {{ args }}

# Helper for viking_run for mpi
viking_run_mpi folder tasks="2" nodes="1" dest_folder="manual_tests" *args="":
    just \
        VORTEX_CMD="mpiexec -n {{ tasks }} ./vortex" \
        VIKING_NUM_TASKS={{ tasks }} \
        VIKING_SLURM_ARGS='#SBATCH --nodes={{ nodes }}' \
        VIKING_JOB_TIME={{ VIKING_JOB_TIME }} \
        VIKING_MODULE=mpi/OpenMPI/4.1.1-GCC-11.2.0 \
        viking_run "mpi" "{{ folder }}" {{dest_folder}} {{ args }}

# View the viking job queue
viking_queue: (viking_ssh "squeue -u " + YORK_USER)
alias vq := viking_queue

# Run all benches
viking_bench_run jump="500" max="5000" targets=TARGETS omp_cpus="20" mpi_tasks="9" mpi_dims="-X 3 -Y 3":
    #!/bin/env hush
    let targets = std.split("{{ targets }}", " ")
    for size in std.range({{ jump }}, {{ max }}, {{ jump }}) do
        if std.contains(targets, "original") then
            { just VIKING_JOB_TIME={{ VIKING_JOB_TIME }}
                    viking_run original /tmp/hipc_original_${size} -x $size -y $size -n }
        end
        if std.contains(targets, "openmp") then
            { just VIKING_JOB_TIME={{ VIKING_JOB_TIME }}
                    viking_run_openmp /tmp/hipc_openmp_${size} {{ omp_cpus }}
                        -x $size -y $size -n }
        end
        if std.contains(targets, "cuda") then
            { just VIKING_JOB_TIME={{ VIKING_JOB_TIME }}
                    viking_run_cuda /tmp/hipc_cuda_${size} -x $size -y $size -n }
        end
        if std.contains(targets, "mpi") then
            { just VIKING_JOB_TIME={{ VIKING_JOB_TIME }}
                    viking_run_mpi /tmp/hipc_mpi_${size} {{ mpi_tasks }} {{ mpi_dims }}
                        -x $size -y $size -n }
        end
    end

# Retrieve slurm logs from `viking_bench_run`
viking_bench_retrieve jump="500" max="5000" targets=TARGETS:
    #!/bin/env hush
    let cwd = std.cwd()
    for size in std.range({{ jump }}, {{ max }}, {{ jump }}) do
        for target in std.iter(std.split("{{ targets }}", " ")) do
            {
                mkdir -pv "{{ VIKING_BENCH_RESULTS_DIR }}/${target}_${size}";
                just viking_rsync_from "scratch/hipc_${target}_${size}/slurm-*"
                    "{{ VIKING_BENCH_RESULTS_DIR }}/${target}_${size}";
                cd "{{ VIKING_BENCH_RESULTS_DIR }}/${target}_${size}";
                nomino -pw -r 'slurm-[0-9]+\\.out' 'slurm.out';
                cd $cwd;
            }
        end
    end
    std.print("========================================\nResults saved to {{ VIKING_BENCH_RESULTS_DIR }}\n")

# Cancel all viking jobs
viking_cancel: && (viking_ssh "scancel -u " + YORK_USER)
    #!/bin/bash
    set -euo pipefail

    read -p "Are you sure you want to cancel all viking jobs for {{ YORK_USER }}? [y/N]" -n 1 -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        echo
        echo "No confirmation from user, exiting..."
        exit 1
    fi
    echo
    echo "Cancelling all viking jobs for {{ YORK_USER }}..."

# Cancel viking jobs and clean up
viking_clean: (viking_cancel) (viking_ssh "rm -rfv ~/scratch/hipc*")
