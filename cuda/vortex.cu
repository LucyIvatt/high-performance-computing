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
#include "kernels.cuh"
#include "args.h"

struct timespec timer;


double get_time()
{
    clock_gettime(CLOCK_MONOTONIC, &timer);
    return (double)(timer.tv_sec + timer.tv_nsec / 1000000000.0);
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
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((imax_h + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
				   (jmax_h + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

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
    cudaDeviceSynchronize();

    if (verbose)
        print_opts();

    allocate_arrays();
    
    problem_set_up<<<1,1>>>(u, v, p, flag);
    cudaDeviceSynchronize();

    apply_boundary_conditions<<<1,1>>>(u, v, p, rhs, f, g, flag);
    cudaDeviceSynchronize();

    setup_time = get_time() - setup_time;

    /* Main loop */
    int iters = 0;
    double t;
    for (t = 0.0; t < t_end_h; t += del_t_h, iters++)
    {
        if (!fixed_dt)
            set_timestep_interval<<<1,1>>>(u, v, p, rhs, f, g, flag);
            cudaDeviceSynchronize();
            cudaMemcpyFromSymbol(&del_t_h, del_t, sizeof(double));

        tentative_velocity_start = get_time();
        compute_tentative_velocity<<<numBlocks, threadsPerBlock>>>(u, v, p, rhs, f, g, flag);
        cudaDeviceSynchronize();
        tentative_velocity_time += get_time() - tentative_velocity_start;

        rhs_start = get_time();
        compute_rhs<<<1,1>>>(u, v, p, rhs, f, g, flag);
        cudaDeviceSynchronize();
        rhs_time += get_time() - rhs_start;

        poisson_start = get_time();
        
        poisson();

        cudaDeviceSynchronize();
        poisson_time += get_time() - poisson_start;

        update_velocity_start = get_time();
        update_velocity<<<numBlocks, threadsPerBlock>>>(u, v, p, rhs, f, g, flag);
        cudaDeviceSynchronize();
        update_velocity_time += get_time() - update_velocity_start;

        apply_boundary_conditions_start = get_time();
        apply_boundary_conditions<<<1,1>>>(u, v, p, rhs, f, g, flag);
        cudaDeviceSynchronize();
        apply_boundary_conditions_time += get_time() - apply_boundary_conditions_start;

        if ((iters % output_freq == 0))
        {
            printf("Step %8d, Time: %14.8e (del_t: %14.8e), Residual: %14.8e\n", iters, t + del_t_h, del_t_h, residual_h);

            if ((!no_output) && (enable_checkpoints))
                write_checkpoint(iters, t + del_t_h);
        }
    } /* End of main loop */
 
    update_host_arrays();

    total_time = get_time() - total_time;

    printf("Step %8d, Time: %14.8e, Residual: %14.8e\n", iters, t, residual_h);
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