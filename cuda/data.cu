#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include "vtk.h"
#include "data.h"

double xlength = 4.0; /* Width of simulated domain */
double ylength = 1.0; /* Height of simulated domain */
int imax = 512;		  /* Number of cells horizontally */
int jmax = 128;		  /* Number of cells vertically */

double t_end = 5.0;	  /* Simulation runtime */
double del_t = 0.003; /* Duration of each timestep */
double tau = 0.5;	  /* Safety factor for timestep control */

int itermax = 100;	/* Maximum number of iterations in SOR */
double eps = 0.001; /* Stopping error threshold for SOR */
double omega = 1.7; /* Relaxation parameter for SOR */
double y = 0.9;		/* Gamma, Upwind differencing factor in PDE discretisation */

double Re = 500.0; /* Reynolds number */
double ui = 1.0;   /* Initial X velocity */
double vi = 0.0;   /* Initial Y velocity */

double delx, dely;

int fluid_cells = 0;

// Grids used for veclocities, pressure, rhs, flag and temporary f and g arrays
int u_size_x, u_size_y;
double *u_h;
struct Array2D u_array_h, u_array_d;

int v_size_x, v_size_y;
double *v_h;
struct Array2D v_array_h, v_array_d;

int p_size_x, p_size_y;
double *p_h;
struct Array2D p_array_h, p_array_d;

int rhs_size_x, rhs_size_y;
double *rhs_h;
struct Array2D rhs_array_h, rhs_array_d;

int f_size_x, f_size_y;
double *f_h;
struct Array2D f_array_h, f_array_d;

int g_size_x, g_size_y;
double *g_h;
struct Array2D g_array_h, g_array_d;

int flag_size_x, flag_size_y;
char *flag_h;
struct Array2D flag_array_h, flag_array_d;



/**
 * @brief Allocate a 2D char array that is addressable using square brackets
 *
 * @param m The first dimension of the array
 * @param n The second dimension of the array
 * @return char** A 2D array
 */
char *alloc_2d_char_array(int m, int n)
{
	return (char *)calloc(m * n, sizeof(char));
}

/**
 * @brief Allocate a 2D array that is addressable using square brackets
 *
 * @param m The first dimension of the array
 * @param n The second dimension of the array
 * @return double* A vectorized 2D array
 */
double* alloc_2d_array(int m, int n)
{
	return (double *)calloc(m * n, sizeof(double));
}

/**

 */
double *copy_double_array_to_device(int m, int n, double *src)
{
	double* dev;
	cudaMalloc(&dev, m * n * sizeof(double));
	cudaMemcpy(dev, src, m * n * sizeof(double), cudaMemcpyHostToDevice);
	return dev;
}

/**

 */
void *copy_double_array_to_host(int m, int n, double *src, double *dest)
{
	cudaMemcpy(dest, src, m * n * sizeof(double), cudaMemcpyDeviceToHost);
}

/**

 * @return char** A 2D array
 */
char *copy_char_array_to_device(int m, int n, char **src)
{

}



/**
 * @brief Free a 2D array
 *
 * @param array The 2D array to free
 */
void free_2d_array_device(void **array)
{

}

/**
 * @brief Free a 2D array
 *
 * @param array The 2D array to free
 */
void free_2d_array_host(void **array)
{


}

