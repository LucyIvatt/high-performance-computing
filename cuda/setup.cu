#include <stdio.h>
#include <stdlib.h>

#include "data.h"
#include "vtk.h"
#include "boundary.h"

/**
 * @brief Set up some default values before arguments are parsed.
 *
 */
void set_defaults()
{
    set_default_base();
}

/**
 * @brief Set up some values after arguments have been parsed.
 *
 */
void setup()
{
    delx_h = xlength / imax_h;
    dely_h = ylength / jmax_h;

    cudaMemcpyToSymbol(delx, &delx_h, sizeof(double));
    cudaMemcpyToSymbol(dely, &dely_h, sizeof(double));

    cudaMemcpyToSymbol(imax, &imax_h, sizeof(int));
    cudaMemcpyToSymbol(jmax, &jmax_h, sizeof(int));
    cudaMemcpyToSymbol(t_end, &t_end_h, sizeof(double));
    cudaMemcpyToSymbol(del_t, &del_t_h, sizeof(double));
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

/**
 * @brief Initialise the velocity arrays and then initialize the flag array,
 * marking any obstacle cells and the edge cells as boundaries. The cells
 * adjacent to boundary cells have their relevant flags set too.
 */
__global__ void problem_set_up(double* u, double* v, double* p, char* flag)
{
    for (int i = 0; i < imax + 2; i++)
    {
        for (int j = 0; j < jmax + 2; j++)
        {
            u[ind(i, j, u_size_y)] = ui;
            v[ind(i, j, v_size_y)] = vi;
            p[ind(i, j, p_size_y)] = 0.0;
        }
    }

    /* Mark a circular obstacle as boundary cells, the rest as fluid */
    double mx = 20.0 / 41.0 * jmax * dely;
    double my = mx;
    double rad1 = 5.0 / 41.0 * jmax * dely;
    for (int i = 1; i <= imax; i++)
    {
        for (int j = 1; j <= jmax; j++)
        {
            double x = (i - 0.5) * delx - mx;
            double y = (j - 0.5) * dely - my;
            flag[ind(i, j, flag_size_y)] = (x * x + y * y <= rad1 * rad1) ? C_B : C_F;
        }
    }

    /* Mark the north & south boundary cells */
    for (int i = 0; i <= imax + 1; i++)
    {
        flag[ind(i, 0, flag_size_y)] = C_B;
        flag[ind(i, jmax + 1, flag_size_y)] = C_B;
    }
    /* Mark the east and west boundary cells */
    for (int j = 1; j <= jmax; j++)
    {
        flag[ind(0, j, flag_size_y)] = C_B;
        flag[ind(imax + 1, j, flag_size_y)] = C_B;
    }

    fluid_cells = imax * jmax;

    /* flags for boundary cells */
    for (int i = 1; i <= imax; i++)
    {
        for (int j = 1; j <= jmax; j++)
        {
            if (!(flag[ind(i, j, flag_size_y)] & C_F))
            {
                fluid_cells--;
                if (flag[ind(i - 1, j, flag_size_y)] & C_F)
                    flag[ind(i, j, flag_size_y)] |= B_W;
                if (flag[ind(i + 1, j, flag_size_y)] & C_F)
                    flag[ind(i, j, flag_size_y)] |= B_E;
                if (flag[ind(i, j - 1, flag_size_y)] & C_F)
                    flag[ind(i, j, flag_size_y)] |= B_S;
                if (flag[ind(i, j + 1, flag_size_y)] & C_F)
                    flag[ind(i, j, flag_size_y)] |= B_N;
            }
        }
    }
}