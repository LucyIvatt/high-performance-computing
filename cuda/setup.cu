#include <stdio.h>
#include <stdlib.h>

#include "data.h"
#include "vtk.h"
#include "kernels.cuh"

/**
 * @brief Set up some default values before arguments are parsed.
 *
 */
void set_defaults()
{
    set_default_base();
}

/**
 * @brief Set up some values after arguments have been parsed, copies to GPU constants as needed.
 *
 */
void setup()
{
    // Values taken from arg inputs
    cudaMemcpyToSymbol(imax, &imax_h, sizeof(int));
    cudaMemcpyToSymbol(jmax, &jmax_h, sizeof(int));
    cudaMemcpyToSymbol(t_end, &t_end_h, sizeof(double));
    cudaMemcpyToSymbol(del_t, &del_t_h, sizeof(double));

    // Values calculated from arg inputs
    delx_h = xlength / imax_h;
    dely_h = ylength / jmax_h;

    cudaMemcpyToSymbol(delx, &delx_h, sizeof(double));
    cudaMemcpyToSymbol(dely, &dely_h, sizeof(double));

    double rdx2_h = 1.0 / (delx_h * delx_h);
    double rdy2_h = 1.0 / (dely_h * dely_h);
    double omega_h = 1.7;
    double beta_2_h = -omega_h / (2.0 * (rdx2_h + rdy2_h));

    cudaMemcpyToSymbol(rdx2, &rdx2_h, sizeof(double));
    cudaMemcpyToSymbol(rdy2, &rdy2_h, sizeof(double));
    cudaMemcpyToSymbol(beta_2, &beta_2_h, sizeof(double));

    double mx_h = 20.0 / 41.0 * jmax_h * dely_h;
    double rad1_h = 5.0 / 41.0 * jmax_h * dely_h;

    cudaMemcpyToSymbol(mx, &mx_h, sizeof(double));
    cudaMemcpyToSymbol(my, &mx_h, sizeof(double));
    cudaMemcpyToSymbol(rad1, &rad1_h, sizeof(double));

    int fluid_cells_h = imax_h * jmax_h;
    cudaMemcpyToSymbol(fluid_cells, &fluid_cells_h, sizeof(int));
}

/**
 * @brief Allocate all of the arrays used by the computation.
 *
 */
void allocate_arrays()
{
    /* Allocate arrays */
    u_size_x_h = imax_h + 2;
    u_size_y_h = jmax_h + 2;
    u_host = alloc_2d_array(u_size_x_h, u_size_y_h);
    u = copy_2d_array_to_gpu(u_host, u_size_x_h, u_size_y_h);
    cudaMemcpyToSymbol(u_size_x, &u_size_x_h, sizeof(int));
    cudaMemcpyToSymbol(u_size_y, &u_size_y_h, sizeof(int));

    v_size_x_h = imax_h + 2;
    v_size_y_h = jmax_h + 2;
    v_host = alloc_2d_array(v_size_x_h, v_size_y_h);
    v = copy_2d_array_to_gpu(v_host, v_size_x_h, v_size_y_h);
    cudaMemcpyToSymbol(v_size_x, &v_size_x_h, sizeof(int));
    cudaMemcpyToSymbol(v_size_y, &v_size_y_h, sizeof(int));


    f_size_x_h = imax_h + 2;
    f_size_y_h = jmax_h + 2;
    f_host = alloc_2d_array(f_size_x_h, f_size_y_h);
    f = copy_2d_array_to_gpu(f_host, f_size_x_h, f_size_y_h);
    cudaMemcpyToSymbol(f_size_x, &f_size_x_h, sizeof(int));
    cudaMemcpyToSymbol(f_size_y, &f_size_y_h, sizeof(int));


    g_size_x_h = imax_h + 2;
    g_size_y_h = jmax_h + 2;
    g_host = alloc_2d_array(g_size_x_h, g_size_y_h);
    g = copy_2d_array_to_gpu(g_host, g_size_x_h, g_size_y_h);
    cudaMemcpyToSymbol(g_size_x, &g_size_x_h, sizeof(int));
    cudaMemcpyToSymbol(g_size_y, &g_size_y_h, sizeof(int));

    p_size_x_h = imax_h + 2;
    p_size_y_h = jmax_h + 2;
    p_host = alloc_2d_array(p_size_x_h, p_size_y_h);
    p = copy_2d_array_to_gpu(p_host, p_size_x_h, p_size_y_h);
    cudaMemcpyToSymbol(p_size_x, &p_size_x_h, sizeof(int));
    cudaMemcpyToSymbol(p_size_y, &p_size_y_h, sizeof(int));

    rhs_size_x_h = imax_h + 2;
    rhs_size_y_h = jmax_h + 2;
    rhs_host = alloc_2d_array(rhs_size_x_h, rhs_size_y_h);
    rhs = copy_2d_array_to_gpu(rhs_host, rhs_size_x_h, rhs_size_y_h);
    cudaMemcpyToSymbol(rhs_size_x, &rhs_size_x_h, sizeof(int));
    cudaMemcpyToSymbol(rhs_size_y, &rhs_size_y_h, sizeof(int));

    flag_size_x_h = imax_h + 2;
    flag_size_y_h = jmax_h + 2;
    flag_host = alloc_2d_char_array(flag_size_x_h, flag_size_y_h);
    flag = copy_2d_char_array_to_gpu(flag_host, flag_size_x_h, flag_size_y_h);
    cudaMemcpyToSymbol(flag_size_x, &flag_size_x_h, sizeof(int));
    cudaMemcpyToSymbol(flag_size_y, &flag_size_y_h, sizeof(int));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((imax_h + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
				   (jmax_h + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    p0 = allocate_2d_gpu_array(1, 1);
    p0_reductions = allocate_2d_gpu_array(numBlocks.x, numBlocks.y);

    residual = allocate_2d_gpu_array(1, 1);
    residual_reductions = allocate_2d_gpu_array(numBlocks.x, numBlocks.y);

    umax_g = allocate_2d_gpu_array(1, 1);
    vmax_g = allocate_2d_gpu_array(1, 1);

    umax_red = allocate_2d_gpu_array(numBlocks.x, numBlocks.y);
    vmax_red = allocate_2d_gpu_array(numBlocks.x, numBlocks.y);


    if (!u_host || !v_host || !f_host || !g_host || !p_host || !rhs_host || !flag_host)
    {
        fprintf(stderr, "Couldn't allocate memory for matrices.\n");
        exit(1);
    }
}

void update_host_arrays() {
    update_host_array(u_host, u, u_size_x_h, u_size_y_h);
    update_host_array(v_host, v, v_size_x_h, v_size_y_h);
    update_host_array(f_host, f, f_size_x_h, f_size_y_h);
    update_host_array(g_host, g, g_size_x_h, g_size_y_h);
    update_host_array(p_host, p, p_size_x_h, p_size_y_h);
    update_host_array(rhs_host, rhs, rhs_size_x_h, rhs_size_y_h);
    update_host_char_array(flag_host, flag, flag_size_x_h, flag_size_y_h);
}

/**
 * @brief Free all of the arrays used for the computation.
 *
 */
void free_arrays()
{
    free_2d_array((void *)u_host);
    free_2d_array((void *)v_host);
    free_2d_array((void *)f_host);
    free_2d_array((void *)g_host);
    free_2d_array((void *)p_host);
    free_2d_array((void *)rhs_host);
    free_2d_array((void *)flag_host);

    free_gpu_array((void *)u);
    free_gpu_array((void *)v);
    free_gpu_array((void *)f);
    free_gpu_array((void *)g);
    free_gpu_array((void *)p);
    free_gpu_array((void *)rhs);
    free_gpu_array((void *)flag);
}

