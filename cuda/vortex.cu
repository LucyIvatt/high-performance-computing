#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#include "data.h"
#include "vtk.h"
#include "setup.h"
#include "boundary.h"
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
void compute_tentative_velocity()
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

/**
 * @brief Calculate the right hand side of the pressure equation
 *
 */
void compute_rhs()
{
    for (int i = 1; i < imax + 1; i++)
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

/**
 * @brief Red/Black SOR to solve the poisson equation.
 *
 * @return Calculated residual of the computation
 *
 */
double poisson()
{
    double rdx2 = 1.0 / (delx * delx);
    double rdy2 = 1.0 / (dely * dely);
    double beta_2 = -omega / (2.0 * (rdx2 + rdy2));

    double p0 = 0.0;
    /* Calculate sum of squares */
    for (int i = 1; i < imax + 1; i++)
    {
        for (int j = 1; j < jmax + 1; j++)
        {
            if (flag[i][j] & C_F)
            {
                p0 += p[i][j] * p[i][j];
            }
        }
    }

    p0 = sqrt(p0 / fluid_cells);
    if (p0 < 0.0001)
    {
        p0 = 1.0;
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

__global__ void residual_reduction_kernel(){}

/**
 * @brief Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
void update_velocity()
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

/**
 * @brief Set the timestep size so that we satisfy the Courant-Friedrichs-Lewy
 * conditions. Otherwise the simulation becomes unstable.
 */
void set_timestep_interval()
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
    double setup_time = get_time();

    double tentative_velocity_time = 0;
    double rhs_time = 0;
    double poisson_time = 0;
    double update_velocity_time = 0;
    double apply_boundary_conditions_time = 0;

    double tentative_velocity_start;
    double rhs_start;
    double poisson_start;
    double update_velocity_start;
    double apply_boundary_conditions_start;

    set_defaults();
    parse_args(argc, argv);
    setup();

    if (verbose)
        print_opts();

    allocate_arrays();
    problem_set_up();

    double res;

    setup_time = get_time() - setup_time;

    /* Main loop */
    int iters = 0;
    double t;
    for (t = 0.0; t < t_end; t += del_t, iters++)
    {
        if (!fixed_dt)
            set_timestep_interval();

        tentative_velocity_start = get_time();
        compute_tentative_velocity();
        tentative_velocity_time += get_time() - tentative_velocity_start;

        rhs_start = get_time();
        compute_rhs();
        rhs_time += get_time() - rhs_start;

        poisson_start = get_time();

        char **d_flag;
        double **d_p, **d_rhs;
        copy_char_array_to_device(flag, d_flag, f_size_x, f_size_y);
        copy_double_array_to_device(p, d_p, p_size_x, p_size_y);
        copy_double_array_to_device(rhs, d_rhs, rhs_size_x, rhs_size_y);

        res = poisson();
        poisson_time += get_time() - poisson_start;

        update_velocity_start = get_time();
        update_velocity();
        update_velocity_time += get_time() - update_velocity_start;

        apply_boundary_conditions_start = get_time();
        apply_boundary_conditions();
        apply_boundary_conditions_time += get_time() - apply_boundary_conditions_start;

        if ((iters % output_freq == 0))
        {
            printf("Step %8d, Time: %14.8e (del_t: %14.8e), Residual: %14.8e\n", iters, t + del_t, del_t, res);

            if ((!no_output) && (enable_checkpoints))
                write_checkpoint(iters, t + del_t);
        }
    } /* End of main loop */

    total_time = get_time() - total_time;

    printf("Step %8d, Time: %14.8e, Residual: %14.8e\n", iters, t, res);
    printf("Simulation complete.\n");

    fprintf(stderr, "Timing Summary\n");
    fprintf(stderr, " Setup Time: %lf\n", setup_time);
    fprintf(stderr, " Tenatative Velocity Time: %lf\n", tentative_velocity_time);
    fprintf(stderr, " RHS Time: %lf\n", rhs_time);
    fprintf(stderr, " Poisson Time: %lf\n", poisson_time);
    fprintf(stderr, " Update Velocity Time: %lf\n", update_velocity_time);
    fprintf(stderr, " Apply Boundary Conditions Time: %lf\n\n", apply_boundary_conditions_time);
    fprintf(stderr, " Total Time: %lf\n", total_time);

    if (!no_output)
        write_result(iters, t);

    free_arrays();

    return 0;
}