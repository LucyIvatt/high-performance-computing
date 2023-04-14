#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include "vtk.h"
#include "data.h"

int imax_h = 512;		  /* Number of cells horizontally */
int jmax_h = 128;		  /* Number of cells vertically */
double t_end_h = 5.0;	  /* Simulation runtime */

int itermax = 100;  /* Maximum number of iterations in SOR */
double eps = 0.001;

double delx_h, dely_h;
double residual_h;

int arr_size_x_h, arr_size_y_h;

/* CONSTANTS ONLY NEEDED ON HOST*/
double xlength = 4.0; /* Width of simulated domain */
double ylength = 1.0; /* Height of simulated domain */

/* WILL BE PASSED IN AS PARAMETERS TO KERNELS FOR ACCESS */
// Grids used for veclocities, pressure, rhs, flag and temporary f and g arrays
double *u, *u_host;
double *v, *v_host;
double *p, *p_host;
double *rhs, *rhs_host;
double *f, *f_host;
double *g, *g_host;
char *flag, *flag_host;

double* p0;
double* p0_reductions;

double *umax_g, *vmax_g;
double *umax_red, *vmax_red;

double* residual;
double* residual_reductions;

double del_t_h = 0.003; /* Duration of each timestep */

/**
 * @brief Allocate a 2D array that is addressable using square brackets
 *
 * @param m The first dimension of the array
 * @param n The second dimension of the array
 * @return double** A 2D array
 */
double *alloc_2d_array(int m, int n)
{
	return (double *)calloc(m * n, sizeof(double));
}

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

double *copy_2d_array_to_gpu(double *src, int m, int n) {
	double* gpu;
	cudaMalloc(&gpu, m * n * sizeof(double));
	cudaMemcpy(gpu, src, m * n * sizeof(double), cudaMemcpyHostToDevice);
	return gpu;
}

double *allocate_2d_gpu_array(int m, int n) {
	double* gpu;
	cudaMalloc(&gpu, m * n * sizeof(double));
	return gpu;
}

char *copy_2d_char_array_to_gpu(char *src, int m, int n) {
	char* gpu;
	cudaMalloc(&gpu, m * n * sizeof(char));
	cudaMemcpy(gpu, src, m * n * sizeof(char), cudaMemcpyHostToDevice);
	return gpu;
}

void update_host_array(double *host, double *gpu, int m, int n){
	cudaMemcpy(host, gpu, m * n *sizeof(double), cudaMemcpyDeviceToHost);
}

void update_host_char_array(char *host, char *gpu, int m, int n){
	cudaMemcpy(host, gpu, m * n *sizeof(char), cudaMemcpyDeviceToHost);
}

/**
 * @brief Free a 2D array
 *
 * @param array The 2D array to free
 */
void free_2d_array(void *array)
{
	free(array);
}

void free_gpu_array(void *array) {
	cudaFree(array);
}
