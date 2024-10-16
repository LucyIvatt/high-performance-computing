#ifndef KERNELS
#define KERNELS

/* CONSTANTS NEEDED ON BOTH GPU AND HOST DUE TO PARSED ARGUMENTS */
extern __constant__ int arr_size_x, arr_size_y;

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


__global__ void setup_uvp_kernel(double *u, double *v, double *p);
__global__ void setup_flag_kernel(char *flag);

__global__ void boundary_conditions_WE_kernel(double* u, double* v);
__global__ void boundary_conditions_NS_kernel(double* u, double* v);
__global__ void boundary_conditions_noslip_kernel(double *u, double *v, char *flag);
__global__ void apply_boundary_conditions_west_edge_kernel(double *u, double *v);

__global__ void tentative_velocity_update_f_kernel(double *u, double *v, double *f, char *flag);
__global__ void tentative_velocity_update_g_kernel(double *u, double *v, double *g, char *flag);
__global__ void tentative_velocity_g_boundaries_kernel(double *g, double *v);
__global__ void tentative_velocity_f_boundaries_kernel(double *f, double *u);

__global__ void compute_rhs_kernel(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag);

__global__ void update_velocity_u_kernel(double *u, double *v, double *p, double *rhs, double *f, double *g, char *flag);
__global__ void update_velocity_v_kernel(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag);

__global__ void p0_reduction_blocks_kernel(double *p, char *flag, double *global_reductions);
__global__ void p0_reduction_global_kernel(double *global_reductions, double *p0, int num_blocks_x, int num_blocks_y);
__global__ void star_computation_kernel(double *u, double *v, double *p, double *rhs, double *f, double *g, char *flag, int rb);
__global__ void residual_reduction_blocks_kernel(double *p, double *rhs, char *flag, double *global_reductions);
__global__ void residual_reduction_global_kernel(double *global_reductions, double *residual, int num_blocks_x, int num_blocks_y, double *p0);

__global__ void set_timestep_interval_kernel(double* umax, double* vmax);
__global__ void abs_max_reduction_blocks_kernel(double *array, double *global_reductions, int version);
__global__ void abs_max_reduction_global_kernel(double *global_reductions, double *output_val, int num_blocks_x, int num_blocks_y);

#endif