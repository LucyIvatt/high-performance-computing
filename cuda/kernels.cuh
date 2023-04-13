#ifndef KERNELS
#define KERNELS

/* CONSTANTS NEEDED ON BOTH GPU AND HOST DUE TO PARSED ARGUMENTS */
extern __constant__ int u_size_x, u_size_y;
extern __constant__ int v_size_x, v_size_y;
extern __constant__ int p_size_x, p_size_y;
extern __constant__ int flag_size_x, flag_size_y;
extern __constant__ int g_size_x, g_size_y;
extern __constant__ int f_size_x, f_size_y;
extern __constant__ int rhs_size_x, rhs_size_y;

extern __constant__ int imax;		  /* Number of cells horizontally */
extern __constant__ int jmax;		  /* Number of cells vertically */
extern __constant__ double t_end;	  /* Simulation runtime */
extern __constant__ double delx;
extern __constant__ double dely;

extern __constant__ double mx;
extern __constant__ double my;
extern __constant__ double rad1;

extern __constant__ double rdx2, rdy2, beta_2;

extern __device__ int fluid_cells;
extern __device__ double del_t; /* Duration of each timestep */


__global__ void problem_set_up_kernel(double* u, double* v, double* p, char* flag);
__global__ void boundary_conditions_kernel_1(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag);
__global__ void apply_boundary_conditions_2(double *u, double *v, double *p, double *rhs, double *f, double *g, char *flag);

__global__ void compute_tentative_velocity_kernel(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag);
__global__ void compute_rhs_kernel(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag);
__global__ void update_velocity_kernel(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag);
__global__ void set_timestep_interval_kernel(double* umax, double* vmax);

__global__ void p0_reduction_blocks_kernel(double *p, char *flag, double *global_reductions);
__global__ void p0_reduction_global_kernel(double *global_reductions, double *p0, int num_blocks_x, int num_blocks_y);
__global__ void star_computation_kernel(double *u, double *v, double *p, double *rhs, double *f, double *g, char *flag, int rb);
__global__ void residual_reduction_blocks_kernel(double *p, double *rhs, char *flag, double *global_reductions);
__global__ void residual_reduction_global_kernel(double *global_reductions, double *residual, int num_blocks_x, int num_blocks_y, double *p0);

__global__ void abs_max_reduction_blocks_kernel(double *array, double *global_reductions, int version);
__global__ void abs_max_reduction_global_kernel(double *global_reductions, double *output_val, int num_blocks_x, int num_blocks_y);

#endif