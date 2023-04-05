#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include "vtk.h"
#include "data.h"

int imax_h = 512;		  /* Number of cells horizontally */
int jmax_h = 128;		  /* Number of cells vertically */
double t_end_h = 5.0;	  /* Simulation runtime */
double delx_h, dely_h;

int u_size_x_h, u_size_y_h;
int v_size_x_h, v_size_y_h;
int p_size_x_h, p_size_y_h;
int flag_size_x_h, flag_size_y_h;
int g_size_x_h, g_size_y_h;
int f_size_x_h, f_size_y_h;
int rhs_size_x_h, rhs_size_y_h;

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
