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

    arr_size_x = imax + 2;
    arr_size_y = jmax + 2;

    rdx2 = 1.0 / (delx * delx);
    rdy2 = 1.0 / (dely * dely);
    beta_2 = -omega / (2.0 * (rdx2 + rdy2));
}

/**
 * @brief Allocate all of the arrays used by the computation.
 *
 */
void allocate_arrays()
{
    /* Allocate arrays */

    u = alloc_2d_array(arr_size_x, arr_size_y);
    v = alloc_2d_array(arr_size_x, arr_size_y);
    f = alloc_2d_array(arr_size_x, arr_size_y);
    g = alloc_2d_array(arr_size_x, arr_size_y);
    p = alloc_2d_array(arr_size_x, arr_size_y);
    rhs = alloc_2d_array(arr_size_x, arr_size_y);
    flag = alloc_2d_char_array(arr_size_x, arr_size_y);

    if (!u || !v || !f || !g || !p || !rhs || !flag)
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
    free_2d_array((void **)u);
    free_2d_array((void **)v);
    free_2d_array((void **)f);
    free_2d_array((void **)g);
    free_2d_array((void **)p);
    free_2d_array((void **)rhs);
    free_2d_array((void **)flag);
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
            u[i][j] = ui;
            v[i][j] = vi;
            p[i][j] = 0.0;
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
            flag[i][j] = (x * x + y * y <= rad1 * rad1) ? C_B : C_F;
        }
    }

    /* Mark the north & south boundary cells */
    for (int i = 0; i <= imax + 1; i++)
    {
        flag[i][0] = C_B;
        flag[i][jmax + 1] = C_B;
    }
    /* Mark the east and west boundary cells */
    for (int j = 1; j <= jmax; j++)
    {
        flag[0][j] = C_B;
        flag[imax + 1][j] = C_B;
    }

    fluid_cells = imax * jmax;

    /* flags for boundary cells */
    for (int i = 1; i <= imax; i++)
    {
        for (int j = 1; j <= jmax; j++)
        {
            if (!(flag[i][j] & C_F))
            {
                fluid_cells--;
                if (flag[i - 1][j] & C_F)
                    flag[i][j] |= B_W;
                if (flag[i + 1][j] & C_F)
                    flag[i][j] |= B_E;
                if (flag[i][j - 1] & C_F)
                    flag[i][j] |= B_S;
                if (flag[i][j + 1] & C_F)
                    flag[i][j] |= B_N;
            }
        }
    }
}