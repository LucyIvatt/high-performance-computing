# high-performance-computing
![C](https://img.shields.io/badge/c-%2300599C.svg?style=for-the-badge&logo=c&logoColor=white)
![nVIDIA](https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green)

This repository contains 3 different solutions to parallelising a provided simple solver for the [Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations) - used to treat laminar flows of viscous, incompressible fluids

The three technologies used are [OpenMP](https://en.wikipedia.org/wiki/OpenMP), [CUDA](https://en.wikipedia.org/wiki/CUDA) & [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface). 

## Initial Investigation

Initially used `gprof` to identify the most time consuming section of the code. This was identified as the poisson function - accounting for
approximately 96.53% of the program's runtime.

<img width="400" alt="image" src="https://github.com/user-attachments/assets/c1e4ad93-09e5-4deb-bf1c-bcc7ed494b45">

## Validation

A python script `validator.py` was used to validate the outputs of each port by comparing the VTK files to the
original. The script categorized each value in the files as exact, close (±0.02), or wrong. 

### Results

| Port    | Wrong | Close (± 0.02) | Exact   | Cosine Similarity | Valid? |
|---------|-------|----------------|---------|-------------------|--------|
| OpenMP  | 0     | 0              | 267,302 | 100               | ✅      |
| Cuda    | 0     | 0              | 267,302 | 100               | ✅      |
| MPI     | 0     | 0              | 267,302 | 100               | ✅       |


**Example output of validation script:**

```shell
Comparing implementation (original.vtk) to parallel implementation (openmp.vtk):

WRONG: 0/267302 – 0.0000%
CLOSE: 0/267302 – 0.0000%
EXACT: 267302/267302 – 100.0000%

Note: Close values are determined using a tolerance value of 0.02. Percentages are calculated to 4 decimal places.

Cosine Similarity: 100.0
PASS: Both files are an exact match – successful parallel implementation.
```
## Ports
### OpenMP Approach
Various locations were found to include the following pragma examples to share loop iterations between threads:
```C
#pragma omp parallel for collapse(2) reduction(+:p0)
#pragma omp parallel for collapse(2) reduction(+:res)
#pragma omp parallel for collapse(2) reduction(max:umax)
#pragma omp parallel for collapse(2) reduction(max:vmax)
```


### CUDA Approach
<img width="453" alt="image" src="https://github.com/user-attachments/assets/43caf272-fed3-4441-a40e-097a998a0790">

### MPI Approach

<img width="785" alt="image" src="https://github.com/user-attachments/assets/63c5b987-0e80-46b7-bc58-ab45dc2bdc33">

## Benchmarks / Speedup

To ensure consistent conditions, all ports were evaluated on [Viking](https://www.york.ac.uk/it-services/tools/viking/), the University of Yorks super computer. The main evaluations measured the total time for the main loop to complete, using a code timer, across 20 problem sizes. Each size was tested multiple times and averaged to reduce outliers. CUDA experiments were conducted with and without checkpoints to assess overhead. An OpenMP experiment was also run to evaluate the effect of thread count. All profiling was performed on Viking.

### Original Analysis
<img width="750" alt="image" src="https://github.com/user-attachments/assets/97c032ce-68df-4447-8f81-642c732c42c0">

### OpenMP Analysis
<img width="758" alt="image" src="https://github.com/user-attachments/assets/2547df47-8436-4aee-a89d-f0b454c4f795">

### CUDA Analysis
<img width="771" alt="image" src="https://github.com/user-attachments/assets/f90361c8-ff55-43c7-b34a-1ce658dea746">

### MPI Analysis

_Unfortunately while the MPI approach kept the validity of the solution, I was not able to successfully complete the approach mentioned previously, hence the lack of a significant speedup._

<img width="603" alt="image" src="https://github.com/user-attachments/assets/66134cf7-a4e4-404d-bf01-a25ffcd78c77">

### Comparative Analysis
<img width="749" alt="image" src="https://github.com/user-attachments/assets/3cdc6d01-c224-4ec5-a6f2-972d38205e7a">


| Port     | Average Time | Speedup |
|----------|--------------|---------|
| Original | 135.26       | -       |
| OpenMP   | 13.96        | x9.7    |
| Cuda     | 22.84        | x5.95   |
| MPI      | 131.81       | x1.02   |






_All code submitted as part of a masters module at the University of York - High-Performance Parallel and Distributed Systems._
