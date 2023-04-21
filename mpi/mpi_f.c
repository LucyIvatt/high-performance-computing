#include <math.h>
#include <mpi.h>

#include "data.h"

double p0_reduction(int rank, int process_num)
{
    // P0 REDUCTION
    int ROWS_PER_PROCESS = (arr_size_x - 2) / (process_num - 1);
    int ROW_REMAINDER = (arr_size_x - 2) % ROWS_PER_PROCESS;

    int counts[process_num]; // number of elements to compute per process
    int displs[process_num]; // where to access the elements from the arrays

    counts[0] = ROW_REMAINDER * arr_size_y; // root node only processes remainder rows
    displs[0] = arr_size_y;                 // automatically displaced by a row as we only care about inner values for p0

    for (int i = 1; i < process_num; i++)
    {
        counts[i] = ROWS_PER_PROCESS * arr_size_y;
        displs[i] = arr_size_y + (ROW_REMAINDER * arr_size_y) + (ROWS_PER_PROCESS * arr_size_y * (i - 1));
    }

    double *p_rows = (double *)malloc(counts[rank] * sizeof(double));
    char *flag_rows = (char *)malloc(counts[rank] * sizeof(char));

    double local_p0 = 0;

    // Sends the relevant part of the p and flag arrays to the processes
    MPI_Scatterv(p[0], counts, displs, MPI_DOUBLE, p_rows, counts[rank], MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Scatterv(flag[0], counts, displs, MPI_CHAR, flag_rows, counts[rank], MPI_CHAR, ROOT, MPI_COMM_WORLD);

    int num_rows = (rank == 0) ? ROW_REMAINDER : ROWS_PER_PROCESS;

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 1; j < arr_size_y - 1; j++)
        {
            if (flag_rows[ind(i, j)] & C_F)
            {
                local_p0 += p_rows[ind(i, j)] * p_rows[ind(i, j)];
            }
        }
    }

    double p0;
    MPI_Reduce(&local_p0, &p0, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (rank == ROOT)
    {
        p0 = sqrt(p0 / fluid_cells);
        if (p0 < 0.0001)
        {
            p0 = 1.0;
        }
        MPI_Bcast(&p0, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    }

    return p0;
}

void five_point_star(int rank, int process_num, int rb)
{
    for (int i = 1; i < imax + 1; i++)
    {
        for (int j = 1; j < jmax + 1; j++)
        {
            if ((i + j) % 2 == rb)
            {
                if (flag[i][j] == (C_F | B_NSEW))
                {
                    /* five point star for interior fluid cells */
                    p[i][j] = (1.0 - omega) * p[i][j] -
                              beta_2 * ((p[i + 1][j] + p[i - 1][j]) * rdx2 + (p[i][j + 1] + p[i][j - 1]) * rdy2 - rhs[i][j]);
                }
                else if (flag[i][j] & C_F)
                {
                    /* modified star near boundary */

                    double eps_E = ((flag[i + 1][j] & C_F) ? 1.0 : 0.0);
                    double eps_W = ((flag[i - 1][j] & C_F) ? 1.0 : 0.0);
                    double eps_N = ((flag[i][j + 1] & C_F) ? 1.0 : 0.0);
                    double eps_S = ((flag[i][j - 1] & C_F) ? 1.0 : 0.0);

                    double beta_mod = -omega / ((eps_E + eps_W) * rdx2 + (eps_N + eps_S) * rdy2);
                    p[i][j] = (1.0 - omega) * p[i][j] -
                              beta_mod * ((eps_E * p[i + 1][j] + eps_W * p[i - 1][j]) * rdx2 + (eps_N * p[i][j + 1] + eps_S * p[i][j - 1]) * rdy2 - rhs[i][j]);
                }
            }
        }
    }
}
