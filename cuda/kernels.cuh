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

extern __device__ int fluid_cells;
extern __device__ double del_t; /* Duration of each timestep */

__global__ void problem_set_up(double* u, double* v, double* p, char* flag);
__global__ void apply_boundary_conditions(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag);
__global__ void compute_tentative_velocity(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag);
__global__ void compute_rhs(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag);
__global__ void poisson(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag, double* res);
__global__ void update_velocity(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag);
__global__ void set_timestep_interval(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag);

#endif