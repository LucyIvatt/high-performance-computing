#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#include "boundary.h"
#include "data.h"
#include "vtk.h"
#include "setup.h"
#include "args.h"

struct timespec timer;

double get_time()
{
    clock_gettime(CLOCK_MONOTONIC, &timer);
    return (double)(timer.tv_sec + timer.tv_nsec / 1000000000.0);
}

/**
 * @brief Computation of tentative velocity field (f, g)
 *
 */
void compute_tentative_velocity(int rank)
{
    if (rank == ROOT)
    {
        for (int i = 1; i < imax; i++)
        {
            for (int j = 1; j < jmax + 1; j++)
            {
                /* only if both adjacent cells are fluid cells */
                if ((flag[i][j] & C_F) && (flag[i + 1][j] & C_F))
                {
                    double du2dx = ((u[i][j] + u[i + 1][j]) * (u[i][j] + u[i + 1][j]) +
                                    y * fabs(u[i][j] + u[i + 1][j]) * (u[i][j] - u[i + 1][j]) -
                                    (u[i - 1][j] + u[i][j]) * (u[i - 1][j] + u[i][j]) -
                                    y * fabs(u[i - 1][j] + u[i][j]) * (u[i - 1][j] - u[i][j])) /
                                   (4.0 * delx);
                    double duvdy = ((v[i][j] + v[i + 1][j]) * (u[i][j] + u[i][j + 1]) +
                                    y * fabs(v[i][j] + v[i + 1][j]) * (u[i][j] - u[i][j + 1]) -
                                    (v[i][j - 1] + v[i + 1][j - 1]) * (u[i][j - 1] + u[i][j]) -
                                    y * fabs(v[i][j - 1] + v[i + 1][j - 1]) * (u[i][j - 1] - u[i][j])) /
                                   (4.0 * dely);
                    double laplu = (u[i + 1][j] - 2.0 * u[i][j] + u[i - 1][j]) / delx / delx +
                                   (u[i][j + 1] - 2.0 * u[i][j] + u[i][j - 1]) / dely / dely;

                    f[i][j] = u[i][j] + del_t * (laplu / Re - du2dx - duvdy);
                }
                else
                {
                    f[i][j] = u[i][j];
                }
            }
        }
        for (int i = 1; i < imax + 1; i++)
        {
            for (int j = 1; j < jmax; j++)
            {
                /* only if both adjacent cells are fluid cells */
                if ((flag[i][j] & C_F) && (flag[i][j + 1] & C_F))
                {
                    double duvdx = ((u[i][j] + u[i][j + 1]) * (v[i][j] + v[i + 1][j]) +
                                    y * fabs(u[i][j] + u[i][j + 1]) * (v[i][j] - v[i + 1][j]) -
                                    (u[i - 1][j] + u[i - 1][j + 1]) * (v[i - 1][j] + v[i][j]) -
                                    y * fabs(u[i - 1][j] + u[i - 1][j + 1]) * (v[i - 1][j] - v[i][j])) /
                                   (4.0 * delx);
                    double dv2dy = ((v[i][j] + v[i][j + 1]) * (v[i][j] + v[i][j + 1]) +
                                    y * fabs(v[i][j] + v[i][j + 1]) * (v[i][j] - v[i][j + 1]) -
                                    (v[i][j - 1] + v[i][j]) * (v[i][j - 1] + v[i][j]) -
                                    y * fabs(v[i][j - 1] + v[i][j]) * (v[i][j - 1] - v[i][j])) /
                                   (4.0 * dely);
                    double laplv = (v[i + 1][j] - 2.0 * v[i][j] + v[i - 1][j]) / delx / delx +
                                   (v[i][j + 1] - 2.0 * v[i][j] + v[i][j - 1]) / dely / dely;

                    g[i][j] = v[i][j] + del_t * (laplv / Re - duvdx - dv2dy);
                }
                else
                {
                    g[i][j] = v[i][j];
                }
            }
        }

        /* f & g at external boundaries */
        for (int j = 1; j < jmax + 1; j++)
        {
            f[0][j] = u[0][j];
            f[imax][j] = u[imax][j];
        }
        for (int i = 1; i < imax + 1; i++)
        {
            g[i][0] = v[i][0];
            g[i][jmax] = v[i][jmax];
        }
    }
}

/**
 * @brief Calculate the right hand side of the pressure equation
 *
 */
void compute_rhs(int rank, int process_num)
{
    int ROWS_PER_PROCESS = (arr_size_x - 2) / (process_num - 1);
    int ROW_REMAINDER = (arr_size_x - 2) % ROWS_PER_PROCESS;

    if (rank == ROOT)
    {
        // Sends the rows to the processes
        for (int r = 1; r < process_num; r++)
        {
            int start_loc = (ROWS_PER_PROCESS * (r - 1));
            int send_size = arr_size_y * (ROWS_PER_PROCESS + 2);

            MPI_Send(f[start_loc], send_size, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
            MPI_Send(g[start_loc], send_size, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
            MPI_Send(rhs[start_loc], send_size, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
            MPI_Send(flag[start_loc], send_size, MPI_CHAR, r, 0, MPI_COMM_WORLD);
            MPI_Send(&del_t, 1, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
        }

        // Recives the new row and updates rhs
        for (int r = 1; r < process_num; r++)
        {
            int start_loc = 1 + (ROWS_PER_PROCESS * (r - 1));
            int recv_size = arr_size_y * ROWS_PER_PROCESS;
            MPI_Recv(rhs[start_loc], recv_size, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // If any rows left that need computing, computed by host
        if (ROW_REMAINDER > 0)
        {
            int start_loc = 1 + (ROWS_PER_PROCESS * (process_num - 1));

            for (int i = start_loc; i < imax + 1; i++)
            {
                for (int j = 1; j < jmax + 1; j++)
                {

                    if (flag[i][j] & C_F)
                    {
                        /* only for fluid and non-surface cells */
                        rhs[i][j] = ((f[i][j] - f[i - 1][j]) / delx +
                                     (g[i][j] - g[i][j - 1]) / dely) /
                                    del_t;
                    }
                }
            }
        }
    }

    else if (rank > 0)
    {
        int recv_size = arr_size_y * (ROWS_PER_PROCESS + 2);

        double f_buff[recv_size];
        double g_buff[recv_size];
        double rhs_buff_recv[recv_size];

        char flag_buff[recv_size];

        MPI_Recv(f_buff, recv_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(g_buff, recv_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(rhs_buff_recv, recv_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(flag_buff, recv_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&del_t, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int send_size = arr_size_y * ROWS_PER_PROCESS;
        double rhs_buff_send[arr_size_y * ROWS_PER_PROCESS];

        // Calculates the square of the numbers
        for (int i = 0; i < ROWS_PER_PROCESS; i++)
        {
            for (int j = 1; j < arr_size_y - 1; j++)
            {
                if (flag_buff[ind_ret(i, j)] & C_F)
                {
                    /* only for fluid and non-surface cells */
                    rhs_buff_send[ind(i, j)] = ((f_buff[ind_ret(i, j)] - f_buff[ind_ret(i - 1, j)]) / delx +
                                                (g_buff[ind_ret(i, j)] - g_buff[ind_ret(i, j - 1)]) / dely) /
                                               del_t;
                }
            }
            // Sets values at the end of the rows to what they already were
            rhs_buff_send[ind(i, 0)] = rhs_buff_recv[ind_ret(i, 0)];
            rhs_buff_send[ind(i, arr_size_y - 1)] = rhs_buff_recv[ind_ret(i, arr_size_y - 1)];
        }
        MPI_Send(&(rhs_buff_send[0]), send_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Red/Black SOR to solve the poisson equation.
 *
 * @return Calculated residual of the computation
 *
 */
double poisson(int rank, int process_num)
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


    /* Red/Black SOR-iteration */
    int iter;
    double res = 0.0;
    for (iter = 0; iter < itermax; iter++)
    {
        for (int rb = 0; rb < 2; rb++)
        {
            for (int i = 1; i < imax + 1; i++)
            {
                for (int j = 1; j < jmax + 1; j++)
                {
                    if ((i + j) % 2 != rb)
                    {
                        continue;
                    }
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

        /* computation of residual */
        for (int i = 1; i < imax + 1; i++)
        {
            for (int j = 1; j < jmax + 1; j++)
            {
                if (flag[i][j] & C_F)
                {
                    double eps_E = ((flag[i + 1][j] & C_F) ? 1.0 : 0.0);
                    double eps_W = ((flag[i - 1][j] & C_F) ? 1.0 : 0.0);
                    double eps_N = ((flag[i][j + 1] & C_F) ? 1.0 : 0.0);
                    double eps_S = ((flag[i][j - 1] & C_F) ? 1.0 : 0.0);

                    /* only fluid cells */
                    double add = (eps_E * (p[i + 1][j] - p[i][j]) -
                                  eps_W * (p[i][j] - p[i - 1][j])) *
                                     rdx2 +
                                 (eps_N * (p[i][j + 1] - p[i][j]) -
                                  eps_S * (p[i][j] - p[i][j - 1])) *
                                     rdy2 -
                                 rhs[i][j];
                    res += add * add;
                }
            }
        }
        res = sqrt(res / fluid_cells) / p0;

        /* convergence? */
        if (res < eps)
            break;
    }

    return res;
}

/**
 * @brief Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
void update_velocity(int rank)
{
    if (rank == ROOT)
    {
        for (int i = 1; i < imax - 2; i++)
        {
            for (int j = 1; j < jmax - 1; j++)
            {
                /* only if both adjacent cells are fluid cells */
                if ((flag[i][j] & C_F) && (flag[i + 1][j] & C_F))
                {
                    u[i][j] = f[i][j] - (p[i + 1][j] - p[i][j]) * del_t / delx;
                }
            }
        }

        for (int i = 1; i < imax - 1; i++)
        {
            for (int j = 1; j < jmax - 2; j++)
            {
                /* only if both adjacent cells are fluid cells */
                if ((flag[i][j] & C_F) && (flag[i][j + 1] & C_F))
                {
                    v[i][j] = g[i][j] - (p[i][j + 1] - p[i][j]) * del_t / dely;
                }
            }
        }
    }
}

/**
 * @brief Set the timestep size so that we satisfy the Courant-Friedrichs-Lewy
 * conditions. Otherwise the simulation becomes unstable.
 */
void set_timestep_interval(int rank)
{
    if (rank == ROOT)
    {
        /* del_t satisfying CFL conditions */
        if (tau >= 1.0e-10)
        { /* else no time stepsize control */
            double umax = 1.0e-10;
            double vmax = 1.0e-10;

            for (int i = 0; i < imax + 2; i++)
            {
                for (int j = 1; j < jmax + 2; j++)
                {
                    umax = fmax(fabs(u[i][j]), umax);
                }
            }

            for (int i = 1; i < imax + 2; i++)
            {
                for (int j = 0; j < jmax + 2; j++)
                {
                    vmax = fmax(fabs(v[i][j]), vmax);
                }
            }

            double deltu = delx / umax;
            double deltv = dely / vmax;
            double deltRe = 1.0 / (1.0 / (delx * delx) + 1 / (dely * dely)) * Re / 2.0;

            if (deltu < deltv)
            {
                del_t = fmin(deltu, deltRe);
            }
            else
            {
                del_t = fmin(deltv, deltRe);
            }
            del_t = tau * del_t; /* multiply by safety factor */
        }
    }
}

/**
 * @brief The main routine that sets up the problem and executes the solving routines routines
 *
 * @param argc The number of arguments passed to the program
 * @param argv An array of the arguments passed to the program
 * @return int The return value of the application
 */
int main(int argc, char *argv[])
{
    /* Timer Initialisations */
    double total_time = get_time();

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    int process_num; // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &process_num);

    int rank; // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    set_defaults();
    parse_args(argc, argv);
    setup();

    if (rank == ROOT)
    {
        if (verbose)
            print_opts();
    }
    allocate_arrays();

    if (rank == ROOT)
        problem_set_up();

    apply_boundary_conditions(rank);

    double res;

    /* Main loop */
    int iters = 0;
    double t;
    for (t = 0.0; t < t_end; t += del_t, iters++)
    {
        if (!fixed_dt)
            set_timestep_interval(rank);

        compute_tentative_velocity(rank);
        compute_rhs(rank, process_num);
        res = poisson(rank, process_num);
        update_velocity(rank);
        apply_boundary_conditions(rank);

        if (rank == ROOT)
        {

            if ((iters % output_freq == 0))
            {
                printf("Step %8d, Time: %14.8e (del_t: %14.8e), Residual: %14.8e\n", iters, t + del_t, del_t, res);

                if ((!no_output) && (enable_checkpoints))
                    write_checkpoint(iters, t + del_t);
            }
        }
    } /* End of main loop */

    if (rank == ROOT)
    {
        printf("Step %8d, Time: %14.8e, Residual: %14.8e\n", iters, t, res);
        printf("Simulation complete.\n");
        total_time = get_time() - total_time;
        fprintf(stderr, "Total Time: %lf\n", total_time);

        if (!no_output)
            write_result(iters, t);

        free_arrays();
    }

    MPI_Finalize();
    return 0;
}
