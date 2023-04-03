#include "data.h"
#include "boundary.h"
#include "extras.h"

/**
 * @brief Given the boundary conditions defined by the flag matrix, update
 * the u and v velocities. Also enforce the boundary conditions at the
 * edges of the matrix.
 */
void apply_boundary_conditions()
{
    for (int j = 0; j < jmax + 2; j++)
    {
        /* Fluid freely flows in from the west */
        u_h[INDEX_2D(0, j, u_h_array.size_x)] = u_h[INDEX_2D(1, j, u_h_array.size_x)];
        v_h[INDEX_2D(0, j, v_h_array.size_x)] = v_h[INDEX_2D(1, j, v_h_array.size_x)];

        /* Fluid freely flows out to the east */
        u_h[INDEX_2D(imax, j, u_h_array.size_x)] = u_h[INDEX_2D(imax - 1, j, u_h_array.size_x)];
        v_h[INDEX_2D(imax + 1, j, v_h_array.size_x)] = v_h[INDEX_2D(imax, j, v_h_array.size_x)];
    }

    for (int i = 0; i < imax + 2; i++)
    {
        /* The vertical velocity approaches 0 at the north and south
         * boundaries, but fluid flows freely in the horizontal direction */
        v_h[INDEX_2D(i, jmax, v_h_array.size_x)] = 0.0;
        u_h[INDEX_2D(i, jmax + 1, u_h_array.size_x)] = u_h[INDEX_2D(i, jmax, u_h_array.size_x)];

        v_h[INDEX_2D(i, 0, v_h_array.size_x)] = 0.0;
        u_h[INDEX_2D(i, 0, u_h_array.size_x)] = u_h[INDEX_2D(i, 1, u_h_array.size_x)];
    }

    /* Apply no-slip boundary conditions to cells that are adjacent to
     * internal obstacle cells. This forces the u and v velocity to
     * tend towards zero in these cells.
     */
    for (int i = 1; i < imax + 1; i++)
    {
        for (int j = 1; j < jmax + 1; j++)
        {
            if (flag_h[INDEX_2D(i, j, flag_h_array.size_x)] & B_NSEW)
            {
                switch (flag_h[INDEX_2D(i, j, flag_h_array.size_x)])
                {
                case B_N:
                    v_h[INDEX_2D(i, j, v_h_array.size_x)] = 0.0;
                    u_h[INDEX_2D(i, j, u_h_array.size_x)] = -u_h[INDEX_2D(i, j + 1, u_h_array.size_x)];
                    u_h[INDEX_2D(i - 1, j, u_h_array.size_x)] = -u_h[INDEX_2D(i - 1, j + 1, u_h_array.size_x)];
                    break;
                case B_E:
                    u_h[INDEX_2D(i, j, u_h_array.size_x)] = 0.0;
                    v_h[INDEX_2D(i, j, v_h_array.size_x)] = -v_h[INDEX_2D(i + 1, j, v_h_array.size_x)];
                    v_h[INDEX_2D(i, j - 1, v_h_array.size_x)] = -v_h[INDEX_2D(i + 1, j - 1, v_h_array.size_x)];
                    break;
                case B_S:
                    v_h[INDEX_2D(i, j - 1, v_h_array.size_x)] = 0.0;
                    u_h[INDEX_2D(i, j, u_h_array.size_x)] = -u_h[INDEX_2D(i, j - 1, u_h_array.size_x)];
                    u_h[INDEX_2D(i - 1, j, u_h_array.size_x)] = -u_h[INDEX_2D(i - 1, j - 1, u_h_array.size_x)];
                    break;
                case B_W:
                    u_h[INDEX_2D(i - 1, j, u_h_array.size_x)] = 0.0;
                    v_h[INDEX_2D(i, j, v_h_array.size_x)] = -v_h[INDEX_2D(i - 1, j, v_h_array.size_x)];
                    v_h[INDEX_2D(i, j - 1, v_h_array.size_x)] = -v_h[INDEX_2D(i - 1, j - 1, v_h_array.size_x)];
                    break;
                case B_NE:
                    v_h[INDEX_2D(i, j, v_h_array.size_x)] = 0.0;
                    u_h[INDEX_2D(i, j, u_h_array.size_x)] = 0.0;
                    v_h[INDEX_2D(i, j - 1, v_h_array.size_x)] = -v_h[INDEX_2D(i + 1, j - 1, v_h_array.size_x)];
                    u_h[INDEX_2D(i - 1, j, u_h_array.size_x)] = -u_h[INDEX_2D(i - 1, j + 1, u_h_array.size_x)];
                    break;
                case B_SE:
                    v_h[INDEX_2D(i, j - 1, v_h_array.size_x)] = 0.0;
                    u_h[INDEX_2D(i, j, u_h_array.size_x)] = 0.0;
                    v_h[INDEX_2D(i, j, v_h_array.size_x)] = -v_h[INDEX_2D(i + 1, j, v_h_array.size_x)];
                    u_h[INDEX_2D(i - 1, j, u_h_array.size_x)] = -u_h[INDEX_2D(i - 1, j - 1, u_h_array.size_x)];
                    break;
                case B_SW:
                    v_h[INDEX_2D(i, j - 1, v_h_array.size_x)] = 0.0;
                    u_h[INDEX_2D(i - 1, j, u_h_array.size_x)] = 0.0;
                    v_h[INDEX_2D(i, j, v_h_array.size_x)] = -v_h[INDEX_2D(i - 1, j, v_h_array.size_x)];
                    u_h[INDEX_2D(i, j, u_h_array.size_x)] = -u_h[INDEX_2D(i, j - 1, u_h_array.size_x)];
                    break;
                case B_NW:
                    v_h[INDEX_2D(i, j, v_h_array.size_x)] = 0.0;
                    u_h[INDEX_2D(i - 1, j, u_h_array.size_x)] = 0.0;
                    v_h[INDEX_2D(i, j - 1, v_h_array.size_x)] = -v_h[INDEX_2D(i - 1, j - 1, v_h_array.size_x)];
                    u_h[INDEX_2D(i, j, u_h_array.size_x)] = -u_h[INDEX_2D(i, j + 1, u_h_array.size_x)];
                    break;
                }
            }
        }
    }

    /* Finally, fix the horizontal velocity at the  western edge to have
     * a continual flow of fluid into the simulation.
     */
    v_h[INDEX_2D(0, 0, v_h_array.size_x)] = 2 * vi - v_h[INDEX_2D(1, 0, v_h_array.size_x)];
    for (int j = 1; j < jmax + 1; j++)
    {
        u_h[INDEX_2D(0, j, u_h_array.size_x)] = ui;
        v_h[INDEX_2D(0, j, v_h_array.size_x)] = 2 * vi - v_h[INDEX_2D(1, j, v_h_array.size_x)];
    }
}