#ifndef DATA_H
#define DATA_H

#define C_B 0x0000 /* This cell is an obstacle/boundary cell */
#define B_N 0x0001 /* This obstacle cell has a fluid cell to the north */
#define B_S 0x0002 /* This obstacle cell has a fluid cell to the south */
#define B_W 0x0004 /* This obstacle cell has a fluid cell to the west */
#define B_E 0x0008 /* This obstacle cell has a fluid cell to the east */
#define B_NW (B_N | B_W)
#define B_SW (B_S | B_W)
#define B_NE (B_N | B_E)
#define B_SE (B_S | B_E)
#define B_NSEW (B_N | B_S | B_E | B_W)

#define C_F 0x0010 /* This cell is a fluid cell */

/* CONSTANTS ONLY NEEDED ON HOST*/
extern double xlength; /* Width of simulated domain */
extern double ylength; /* Height of simulated domain */

/* CONSTANTS ONLY NEEDED ON THE GPU THAT CAN BE SET WITHOUT PARSING ARGS */

__constant__ double tau = 0.5;	  /* Safety factor for timestep control */
__constant__ int itermax = 100;	/* Maximum number of iterations in SOR */
__constant__ double eps = 0.001; /* Stopping error threshold for SOR */
__constant__ double omega = 1.7; /* Relaxation parameter for SOR */
__constant__ double y = 0.9;		/* Gamma, Upwind differencing factor in PDE discretisation */

__constant__ double Re = 500.0; /* Reynolds number */
__constant__ double ui = 1.0;   /* Initial X velocity */
__constant__ double vi = 0.0;   /* Initial Y velocity */

/* CONSTANTS NEEDED ON BOTH GPU AND HOST DUE TO PARSED ARGUMENTS */
__constant__ int u_size_x, u_size_y;
__constant__ int v_size_x, v_size_y;
__constant__ int p_size_x, p_size_y;
__constant__ int flag_size_x, flag_size_y;
__constant__ int g_size_x, g_size_y;
__constant__ int f_size_x, f_size_y;
__constant__ int rhs_size_x, rhs_size_y;

__constant__ int imax;		  /* Number of cells horizontally */
__constant__ int jmax;		  /* Number of cells vertically */
__constant__ double t_end;	  /* Simulation runtime */
__constant__ double delx;
__constant__ double dely;

extern int imax_h;		  /* Number of cells horizontally */
extern int jmax_h;		  /* Number of cells vertically */
extern double t_end_h;	  /* Simulation runtime */
extern double del_t_h; /* Duration of each timestep */
extern double delx_h, dely_h;

extern int u_size_x_h, u_size_y_h;
extern int v_size_x_h, v_size_y_h;
extern int p_size_x_h, p_size_y_h;
extern int flag_size_x_h, flag_size_y_h;
extern int g_size_x_h, g_size_y_h;
extern int f_size_x_h, f_size_y_h;
extern int rhs_size_x_h, rhs_size_y_h;

/* WILL BE PASSED IN AS PARAMETERS TO KERNELS FOR ACCESS */
// Grids used for veclocities, pressure, rhs, flag and temporary f and g arrays
extern double *u, *u_host;
extern double *v, *v_host;
extern double *p, *p_host;
extern double *rhs, *rhs_host;
extern double *f, *f_host;
extern double *g, *g_host;
extern char *flag, *flag_host;

__device__ int fluid_cells = 0;
__device__ double del_t; /* Duration of each timestep */

#define ind(i, j, m) ((i) * (m) + (j))

double *alloc_2d_array(int m, int n);
char *alloc_2d_char_array(int m, int n);

double *copy_2d_array_to_gpu(double *src, int m, int n);
char *copy_2d_char_array_to_gpu(char *src, int m, int n);

void update_host_array(double *host, double *gpu, int m, int n);
void update_host_char_array(char *host, char *gpu, int m, int n);

void free_2d_array(void *array);
void free_gpu_array(void *array);

#endif