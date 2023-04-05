#include "data.h"
#include "boundary.h"

/**
 * @brief Given the boundary conditions defined by the flag matrix, update
 * the u and v velocities. Also enforce the boundary conditions at the
 * edges of the matrix.
 */
__global__ void apply_boundary_conditions(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag)
{
    for (int j = 0; j < jmax + 2; j++)
    {
        /* Fluid freely flows in from the west */
        u[ind(0, j, u_size_y)] = u[ind(1, j, u_size_y)];
        v[ind(0, j, v_size_y)] = v[ind(1, j, v_size_y)];

        /* Fluid freely flows out to the east */
        u[ind(imax, j, u_size_y)] = u[ind(imax - 1, j, u_size_y)];
        v[ind(imax + 1, j, v_size_y)] = v[ind(imax, j, v_size_y)];
    }

    for (int i = 0; i < imax + 2; i++)
    {
        /* The vertical velocity approaches 0 at the north and south
         * boundaries, but fluid flows freely in the horizontal direction */
        v[ind(i, jmax, v_size_y)] = 0.0;
        u[ind(i, jmax + 1, u_size_y)] = u[ind(i, jmax, u_size_y)];

        v[ind(i, 0, v_size_y)] = 0.0;
        u[ind(i, 0, u_size_y)] = u[ind(i, 1, u_size_y)];
    }

    /* Apply no-slip boundary conditions to cells that are adjacent to
     * internal obstacle cells. This forces the u and v velocity to
     * tend towards zero in these cells.
     */
    for (int i = 1; i < imax + 1; i++)
    {
        for (int j = 1; j < jmax + 1; j++)
        {
            if (flag[ind(i, j, flag_size_y)] & B_NSEW)
            {
                switch (flag[ind(i, j, flag_size_y)])
                {
                case B_N:
                    v[ind(i, j, v_size_y)] = 0.0;
                    u[ind(i, j, u_size_y)] = -u[ind(i, j + 1, u_size_y)];
                    u[ind(i - 1, j, u_size_y)] = -u[ind(i - 1, j + 1, u_size_y)];
                    break;
                case B_E:
                    u[ind(i, j, u_size_y)] = 0.0;
                    v[ind(i, j, v_size_y)] = -v[ind(i + 1, j, v_size_y)];
                    v[ind(i, j - 1, v_size_y)] = -v[ind(i + 1, j - 1, v_size_y)];
                    break;
                case B_S:
                    v[ind(i, j - 1, v_size_y)] = 0.0;
                    u[ind(i, j, u_size_y)] = -u[ind(i, j - 1, u_size_y)];
                    u[ind(i - 1, j, u_size_y)] = -u[ind(i - 1, j - 1, u_size_y)];
                    break;
                case B_W:
                    u[ind(i - 1, j, u_size_y)] = 0.0;
                    v[ind(i, j, v_size_y)] = -v[ind(i - 1, j, v_size_y)];
                    v[ind(i, j - 1, v_size_y)] = -v[ind(i - 1, j - 1, v_size_y)];
                    break;
                case B_NE:
                    v[ind(i, j, v_size_y)] = 0.0;
                    u[ind(i, j, u_size_y)] = 0.0;
                    v[ind(i, j - 1, v_size_y)] = -v[ind(i + 1, j - 1, v_size_y)];
                    u[ind(i - 1, j, u_size_y)] = -u[ind(i - 1, j + 1, u_size_y)];
                    break;
                case B_SE:
                    v[ind(i, j - 1, v_size_y)] = 0.0;
                    u[ind(i, j, u_size_y)] = 0.0;
                    v[ind(i, j, v_size_y)] = -v[ind(i + 1, j, v_size_y)];
                    u[ind(i - 1, j, u_size_y)] = -u[ind(i - 1, j - 1, u_size_y)];
                    break;
                case B_SW:
                    v[ind(i, j - 1, v_size_y)] = 0.0;
                    u[ind(i - 1, j, u_size_y)] = 0.0;
                    v[ind(i, j, v_size_y)] = -v[ind(i - 1, j, v_size_y)];
                    u[ind(i, j, u_size_y)] = -u[ind(i, j - 1, u_size_y)];
                    break;
                case B_NW:
                    v[ind(i, j, v_size_y)] = 0.0;
                    u[ind(i - 1, j, u_size_y)] = 0.0;
                    v[ind(i, j - 1, v_size_y)] = -v[ind(i - 1, j - 1, v_size_y)];
                    u[ind(i, j, u_size_y)] = -u[ind(i, j + 1, u_size_y)];
                    break;
                }
            }
        }
    }

    /* Finally, fix the horizontal velocity at the  western edge to have
     * a continual flow of fluid into the simulation.
     */
    v[ind(0, 0, v_size_y)] = 2 * vi - v[ind(1, 0, v_size_y)];
    for (int j = 1; j < jmax + 1; j++)
    {
        u[ind(0, j, u_size_y)] = ui;
        v[ind(0, j, v_size_y)] = 2 * vi - v[ind(1, j, v_size_y)];
    }
}