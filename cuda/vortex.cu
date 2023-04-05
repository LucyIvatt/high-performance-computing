#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <time.h>

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
__global__ void compute_tentative_velocity(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag)
{
    for (int i = 1; i < imax; i++)
    {
        for (int j = 1; j < jmax + 1; j++)
        {
            /* only if both adjacent cells are fluid cells */
            if ((flag[ind(i, j, flag_size_y)] & C_F) && (flag[ind(i + 1, j, flag_size_y)] & C_F))
            {
                double du2dx = ((u[ind(i, j, u_size_y)] + u[ind(i + 1, j, u_size_y)]) * (u[ind(i, j, u_size_y)] + u[ind(i + 1, j, u_size_y)]) +
                                y * fabs(u[ind(i, j, u_size_y)] + u[ind(i + 1, j, u_size_y)]) * (u[ind(i, j, u_size_y)] - u[ind(i + 1, j, u_size_y)]) -
                                (u[ind(i - 1, j, u_size_y)] + u[ind(i, j, u_size_y)]) * (u[ind(i - 1, j, u_size_y)] + u[ind(i, j, u_size_y)]) -
                                y * fabs(u[ind(i - 1, j, u_size_y)] + u[ind(i, j, u_size_y)]) * (u[ind(i - 1, j, u_size_y)] - u[ind(i, j, u_size_y)])) /
                               (4.0 * delx);
                double duvdy = ((v[ind(i, j, v_size_y)] + v[ind(i + 1, j, v_size_y)]) * (u[ind(i, j, u_size_y)] + u[ind(i, j + 1, u_size_y)]) +
                                y * fabs(v[ind(i, j, v_size_y)] + v[ind(i + 1, j, v_size_y)]) * (u[ind(i, j, u_size_y)] - u[ind(i, j + 1, u_size_y)]) -
                                (v[ind(i, j - 1, v_size_y)] + v[ind(i + 1, j - 1, v_size_y)]) * (u[ind(i, j - 1, u_size_y)] + u[ind(i, j, u_size_y)]) -
                                y * fabs(v[ind(i, j - 1, v_size_y)] + v[ind(i + 1, j - 1, v_size_y)]) * (u[ind(i, j - 1, u_size_y)] - u[ind(i, j, u_size_y)])) /
                               (4.0 * dely);
                double laplu = (u[ind(i + 1, j, u_size_y)] - 2.0 * u[ind(i, j, u_size_y)] + u[ind(i - 1, j, u_size_y)]) / delx / delx +
                               (u[ind(i, j + 1, u_size_y)] - 2.0 * u[ind(i, j, u_size_y)] + u[ind(i, j - 1, u_size_y)]) / dely / dely;

                f[ind(i, j, f_size_y)] = u[ind(i, j, u_size_y)] + del_t * (laplu / Re - du2dx - duvdy);
            }
            else
            {
                f[ind(i, j, f_size_y)] = u[ind(i, j, u_size_y)];
            }
        }
    }
    for (int i = 1; i < imax + 1; i++)
    {
        for (int j = 1; j < jmax; j++)
        {
            /* only if both adjacent cells are fluid cells */
            if ((flag[ind(i, j, flag_size_y)] & C_F) && (flag[ind(i, j + 1, flag_size_y)] & C_F))
            {
                double duvdx = ((u[ind(i, j, u_size_y)] + u[ind(i, j + 1, u_size_y)]) * (v[ind(i, j, v_size_y)] + v[ind(i + 1, j, v_size_y)]) +
                                y * fabs(u[ind(i, j, u_size_y)] + u[ind(i, j + 1, u_size_y)]) * (v[ind(i, j, v_size_y)] - v[ind(i + 1, j, v_size_y)]) -
                                (u[ind(i - 1, j, u_size_y)] + u[ind(i - 1, j + 1, u_size_y)]) * (v[ind(i - 1, j, v_size_y)] + v[ind(i, j, v_size_y)]) -
                                y * fabs(u[ind(i - 1, j, u_size_y)] + u[ind(i - 1, j + 1, u_size_y)]) * (v[ind(i - 1, j, v_size_y)] - v[ind(i, j, v_size_y)])) /
                               (4.0 * delx);
                double dv2dy = ((v[ind(i, j, v_size_y)] + v[ind(i, j + 1, v_size_y)]) * (v[ind(i, j, v_size_y)] + v[ind(i, j + 1, v_size_y)]) +
                                y * fabs(v[ind(i, j, v_size_y)] + v[ind(i, j + 1, v_size_y)]) * (v[ind(i, j, v_size_y)] - v[ind(i, j + 1, v_size_y)]) -
                                (v[ind(i, j - 1, v_size_y)] + v[ind(i, j, v_size_y)]) * (v[ind(i, j - 1, v_size_y)] + v[ind(i, j, v_size_y)]) -
                                y * fabs(v[ind(i, j - 1, v_size_y)] + v[ind(i, j, v_size_y)]) * (v[ind(i, j - 1, v_size_y)] - v[ind(i, j, v_size_y)])) /
                               (4.0 * dely);
                double laplv = (v[ind(i + 1, j, v_size_y)] - 2.0 * v[ind(i, j, v_size_y)] + v[ind(i - 1, j, v_size_y)]) / delx / delx +
                               (v[ind(i, j + 1, v_size_y)] - 2.0 * v[ind(i, j, v_size_y)] + v[ind(i, j - 1, v_size_y)]) / dely / dely;

                g[ind(i, j, g_size_y)] = v[ind(i, j, v_size_y)] + del_t * (laplv / Re - duvdx - dv2dy);
            }
            else
            {
                g[ind(i, j, g_size_y)] = v[ind(i, j, v_size_y)];
            }
        }
    }

    /* f & g at external boundaries */
    for (int j = 1; j < jmax + 1; j++)
    {
        f[ind(0, j, f_size_y)] = u[ind(0, j, u_size_y)];
        f[ind(imax, j, f_size_y)] = u[ind(imax, j, u_size_y)];
    }
    for (int i = 1; i < imax + 1; i++)
    {
        g[ind(i, 0, g_size_y)] = v[ind(i, 0, v_size_y)];
        g[ind(i, jmax, g_size_y)] = v[ind(i, jmax, v_size_y)];
    }
}

/**
 * @brief Calculate the right hand side of the pressure equation
 *
 */
__global__ void compute_rhs(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag)
{
    for (int i = 1; i < imax + 1; i++)
    {
        for (int j = 1; j < jmax + 1; j++)
        {
            if (flag[ind(i, j, flag_size_y)] & C_F)
            {
                /* only for fluid and non-surface cells */
                rhs[ind(i, j, rhs_size_y)] = ((f[ind(i, j, f_size_y)] - f[ind(i - 1, j, f_size_y)]) / delx +
                             (g[ind(i, j, g_size_y)] - g[ind(i, j - 1, g_size_y)]) / dely) /
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
__global__ void poisson(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag, double* res)
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
            if (flag[ind(i, j, flag_size_y)] & C_F)
            {
                p0 += p[ind(i, j, p_size_y)] * p[ind(i, j, p_size_y)];
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
                    if (flag[ind(i, j, flag_size_y)] == (C_F | B_NSEW))
                    {
                        /* five point star for interior fluid cells */
                        p[ind(i, j, p_size_y)] = (1.0 - omega) * p[ind(i, j, p_size_y)] -
                                  beta_2 * ((p[ind(i + 1, j, p_size_y)] + p[ind(i - 1, j, p_size_y)]) * rdx2 + (p[ind(i, j + 1, p_size_y)] + p[ind(i, j - 1, p_size_y)]) * rdy2 - rhs[ind(i, j, rhs_size_y)]);
                    }
                    else if (flag[ind(i, j, flag_size_y)] & C_F)
                    {
                        /* modified star near boundary */

                        double eps_E = ((flag[ind(i + 1, j, flag_size_y)] & C_F) ? 1.0 : 0.0);
                        double eps_W = ((flag[ind(i - 1, j, flag_size_y)] & C_F) ? 1.0 : 0.0);
                        double eps_N = ((flag[ind(i, j + 1, flag_size_y)] & C_F) ? 1.0 : 0.0);
                        double eps_S = ((flag[ind(i, j - 1, flag_size_y)] & C_F) ? 1.0 : 0.0);

                        double beta_mod = -omega / ((eps_E + eps_W) * rdx2 + (eps_N + eps_S) * rdy2);
                        p[ind(i, j, p_size_y)] = (1.0 - omega) * p[ind(i, j, p_size_y)] -
                                  beta_mod * ((eps_E * p[ind(i + 1, j, p_size_y)] + eps_W * p[ind(i - 1, j, p_size_y)]) * rdx2 + (eps_N * p[ind(i, j + 1, p_size_y)] + eps_S * p[ind(i, j - 1, p_size_y)]) * rdy2 - rhs[ind(i, j, rhs_size_y)]);
                    }
                }
            }
        }

        /* computation of residual */
        for (int i = 1; i < imax + 1; i++)
        {
            for (int j = 1; j < jmax + 1; j++)
            {
                if (flag[ind(i, j, flag_size_y)] & C_F)
                {
                    double eps_E = ((flag[ind(i + 1, j, flag_size_y)] & C_F) ? 1.0 : 0.0);
                    double eps_W = ((flag[ind(i - 1, j, flag_size_y)] & C_F) ? 1.0 : 0.0);
                    double eps_N = ((flag[ind(i, j + 1, flag_size_y)] & C_F) ? 1.0 : 0.0);
                    double eps_S = ((flag[ind(i, j - 1, flag_size_y)] & C_F) ? 1.0 : 0.0);

                    /* only fluid cells */
                    double add = (eps_E * (p[ind(i + 1, j, p_size_y)] - p[ind(i, j, p_size_y)]) -
                                  eps_W * (p[ind(i, j, p_size_y)] - p[ind(i - 1, j, p_size_y)])) *
                                     rdx2 +
                                 (eps_N * (p[ind(i, j + 1, p_size_y)] - p[ind(i, j, p_size_y)]) -
                                  eps_S * (p[ind(i, j, p_size_y)] - p[ind(i, j - 1, p_size_y)])) *
                                     rdy2 -
                                 rhs[ind(i, j, rhs_size_y)];
                    *res += add * add;
                }
            }
        }
        *res = sqrt(*res / fluid_cells) / p0;

        /* convergence? */
        if (*res < eps)
            break;
    }
}

/**
 * @brief Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
__global__ void update_velocity(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag)
{
    for (int i = 1; i < imax - 2; i++)
    {
        for (int j = 1; j < jmax - 1; j++)
        {
            /* only if both adjacent cells are fluid cells */
            if ((flag[ind(i, j, flag_size_y)] & C_F) && (flag[ind(i + 1, j, flag_size_y)] & C_F))
            {
                u[ind(i, j, u_size_y)] = f[ind(i, j, f_size_y)] - (p[ind(i + 1, j, p_size_y)] - p[ind(i, j, p_size_y)]) * del_t / delx;
            }
        }
    }

    for (int i = 1; i < imax - 1; i++)
    {
        for (int j = 1; j < jmax - 2; j++)
        {
            /* only if both adjacent cells are fluid cells */
            if ((flag[ind(i, j, flag_size_y)] & C_F) && (flag[ind(i, j + 1, flag_size_y)] & C_F))
            {
                v[ind(i, j, v_size_y)] = g[ind(i, j, g_size_y)] - (p[ind(i, j + 1, p_size_y)] - p[ind(i, j, p_size_y)]) * del_t / dely;
            }
        }
    }
}

/**
 * @brief Set the timestep size so that we satisfy the Courant-Friedrichs-Lewy
 * conditions. Otherwise the simulation becomes unstable.
 */
__global__ void set_timestep_interval(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag)
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
                umax = fmax(fabs(u[ind(i, j, u_size_y)]), umax);
            }
        }

        for (int i = 1; i < imax + 2; i++)
        {
            for (int j = 0; j < jmax + 2; j++)
            {
                vmax = fmax(fabs(v[ind(i, j, v_size_y)]), vmax);
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
    
    problem_set_up<<<1,1>>>(u, v, p, flag);
    apply_boundary_conditions<<<1,1>>>(u, v, p, rhs, f, g, flag);

    double res=0;

    setup_time = get_time() - setup_time;

    /* Main loop */
    int iters = 0;
    double t;
    for (t = 0.0; t < t_end; t += del_t, iters++)
    {
        if (!fixed_dt)
            set_timestep_interval<<<1,1>>>(u, v, p, rhs, f, g, flag);

        tentative_velocity_start = get_time();
        compute_tentative_velocity<<<1,1>>>(u, v, p, rhs, f, g, flag);
        tentative_velocity_time += get_time() - tentative_velocity_start;

        rhs_start = get_time();
        compute_rhs<<<1,1>>>(u, v, p, rhs, f, g, flag);
        rhs_time += get_time() - rhs_start;

        poisson_start = get_time();
        poisson<<<1,1>>>(u, v, p, rhs, f, g, flag, &res);
        poisson_time += get_time() - poisson_start;

        update_velocity_start = get_time();
        update_velocity<<<1,1>>>(u, v, p, rhs, f, g, flag);
        update_velocity_time += get_time() - update_velocity_start;

        apply_boundary_conditions_start = get_time();
        apply_boundary_conditions<<<1,1>>>(u, v, p, rhs, f, g, flag);
        apply_boundary_conditions_time += get_time() - apply_boundary_conditions_start;

        if ((iters % output_freq == 0))
        {
            printf("Step %8d, Time: %14.8e (del_t: %14.8e), Residual: %14.8e\n", iters, t + del_t, del_t, res);

            if ((!no_output) && (enable_checkpoints))
                write_checkpoint(iters, t + del_t);
        }
    } /* End of main loop */

    update_host_arrays();

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