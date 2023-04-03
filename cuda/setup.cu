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
    delx = xlength / imax;
    dely = ylength / jmax;
}

/**
 * @brief Allocate all of the arrays used by the computation.
 *
 */
void allocate_arrays()
{
    /* Allocate arrays */
    u_size_x = imax + 2;
    u_size_y = jmax + 2;
    u_h = alloc_2d_array(u_size_x, u_size_y);
    u_h_array = {.data=u_h, .size_x=u_size_x, .size_y=u_size_y};


    v_size_x = imax + 2;
    v_size_y = jmax + 2;
    v_h = alloc_2d_array(v_size_x, v_size_y);
    v_h_array = {.data=v_h, .size_x=v_size_x, .size_y=v_size_y};


    f_size_x = imax + 2;
    f_size_y = jmax + 2;
    f_h = alloc_2d_array(f_size_x, f_size_y);
    f_h_array = {.data=f_h, .size_x=f_size_x, .size_y=f_size_y};


    g_size_x = imax + 2;
    g_size_y = jmax + 2;
    g_h = alloc_2d_array(g_size_x, g_size_y);
    g_h_array = {.data=g_h, .size_x=g_size_x, .size_y=g_size_y};


    p_size_x = imax + 2;
    p_size_y = jmax + 2;
    p_h = alloc_2d_array(p_size_x, p_size_y);
    p_h_array = {.data=p_h, .size_x=p_size_x, .size_y=p_size_y};


    rhs_size_x = imax + 2;
    rhs_size_y = jmax + 2;
    rhs_h = alloc_2d_array(rhs_size_x, rhs_size_y);
    rhs_h_array = {.data=rhs_h, .size_x=rhs_size_x, .size_y=rhs_size_y};


    flag_size_x = imax + 2;
    flag_size_y = jmax + 2;
    flag_h = alloc_2d_char_array(flag_size_x, flag_size_y);
    flag_h_array = {.data=flag_h, .size_x=flag_size_x, .size_y=flag_size_y};


    if (!u_h || !v_h || !f_h || !g_h || !p_h || !rhs_h || !flag_h)
    {
        fprintf(stderr, "Couldn't allocate memory for matrices.\n");
        exit(1);
    }
}

/**
 * @brief Free all of the arrays used for the computation.
 *
 */
void free_arrays()
{
    free_2d_array_host((void **)u_h);
    free_2d_array_host((void **)v_h);
    free_2d_array_host((void **)f_h);
    free_2d_array_host((void **)g_h);
    free_2d_array_host((void **)p_h);
    free_2d_array_host((void **)rhs_h);
    free_2d_array_host((void **)flag_h);
}

/**
 * @brief Initialise the velocity arrays and then initialize the flag array,
 * marking any obstacle cells and the edge cells as boundaries. The cells
 * adjacent to boundary cells have their relevant flags set too.
 */
void problem_set_up()
{
    for (int i = 0; i < imax + 2; i++)
    {
        for (int j = 0; j < jmax + 2; j++)
        {
            u_h[i][j] = ui;
            v_h[i][j] = vi;
            p_h[i][j] = 0.0;
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
            flag_h[i][j] = (x * x + y * y <= rad1 * rad1) ? C_B : C_F;
        }
    }

    /* Mark the north & south boundary cells */
    for (int i = 0; i <= imax + 1; i++)
    {
        flag_h[i][0] = C_B;
        flag_h[i][jmax + 1] = C_B;
    }
    /* Mark the east and west boundary cells */
    for (int j = 1; j <= jmax; j++)
    {
        flag_h[0][j] = C_B;
        flag_h[imax + 1][j] = C_B;
    }

    fluid_cells = imax * jmax;

    /* flags for boundary cells */
    for (int i = 1; i <= imax; i++)
    {
        for (int j = 1; j <= jmax; j++)
        {
            if (!(flag_h[i][j] & C_F))
            {
                fluid_cells--;
                if (flag_h[i - 1][j] & C_F)
                    flag_h[i][j] |= B_W;
                if (flag_h[i + 1][j] & C_F)
                    flag_h[i][j] |= B_E;
                if (flag_h[i][j - 1] & C_F)
                    flag_h[i][j] |= B_S;
                if (flag_h[i][j + 1] & C_F)
                    flag_h[i][j] |= B_N;
            }
        }
    }

    apply_boundary_conditions();
}