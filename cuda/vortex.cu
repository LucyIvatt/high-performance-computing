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

void boundary_conditions(dim3 threads, dim3 blocks) {
    // Creates streams to run kernels in parallel
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    // Runs kernels WE and NS in parallel
    boundary_conditions_WE_kernel<<<blocks, threads, 0, s1>>>(u, v);
    boundary_conditions_NS_kernel<<<blocks, threads, 0, s2>>>(u, v);

    // Ensures last two kernels have completed
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);

    // Runs noslip and boundary condition kernels in parallel
    boundary_conditions_noslip_kernel<<<blocks, threads, 0, s1>>>(u, v, flag);
    apply_boundary_conditions_west_edge_kernel<<<blocks, threads, 0, s2>>>(u, v);

    // Closes additional streams when completed
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);

    // Makes function inline with default stream
    cudaDeviceSynchronize();
}

void timestep_interval(dim3 threads, dim3 blocks, int reduction_threads){
    // Creates streams to run kernels in parallel
    abs_max_reduction_blocks_kernel<<<blocks, threads, threads.x * threads.y * sizeof(double)>>>(u, umax_red, 0);
    abs_max_reduction_blocks_kernel<<<blocks, threads, threads.x * threads.y * sizeof(double)>>>(v, vmax_red, 1);
    abs_max_reduction_global_kernel<<<1, reduction_threads, reduction_threads * sizeof(double)>>>(umax_red, umax_g, blocks.x, blocks.y);
    abs_max_reduction_global_kernel<<<1, reduction_threads, reduction_threads * sizeof(double)>>>(vmax_red, vmax_g, blocks.x, blocks.y);

    // Completes the final sequential part of 
    set_timestep_interval_kernel<<<1, 1>>>(umax_g, vmax_g);
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&del_t_h, del_t, sizeof(double));
}

void compute_tentative_velocity(dim3 threads, dim3 blocks){
    cudaStream_t s1, s2, s3, s4;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    cudaStreamCreate(&s3);
    cudaStreamCreate(&s4);
    
    tentative_velocity_update_f_kernel<<<blocks, threads, 0, s1>>>(u, v, f, flag);
    tentative_velocity_update_g_kernel<<<blocks, threads, 0, s2>>>(u, v, g, flag);
    tentative_velocity_g_boundaries_kernel<<<blocks, threads, 0, s3>>>(g, v);
    tentative_velocity_f_boundaries_kernel<<<blocks, threads, 0, s4>>>(f, u);
    cudaDeviceSynchronize();

    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaStreamDestroy(s3);
    cudaStreamDestroy(s4);
    
    // Makes function inline with default stream
    cudaDeviceSynchronize();
    }

void compute_rhs(dim3 threads, dim3 blocks) {
    compute_rhs_kernel<<<blocks, threads>>>(u, v, p, rhs, f, g, flag);
}

void poisson(dim3 threads, dim3 blocks, int reduction_threads)
{
    /* p0 Reduction*/
    p0_reduction_blocks_kernel<<<blocks, threads, threads.x * threads.y * sizeof(double)>>>(p, flag, p0_reductions);
    p0_reduction_global_kernel<<<1, threads, reduction_threads * sizeof(double)>>>(p0_reductions, p0, blocks.x, blocks.y);

    /* Red/Black SOR-iteration */
    for (int iter = 0; iter < itermax; iter++)
    {
        // Star computation for even indicies then odd indicies
        star_computation_kernel<<<blocks, threads>>>(u, v, p, rhs, f, g, flag, 0);
        star_computation_kernel<<<blocks, threads>>>(u, v, p, rhs, f, g, flag, 1);

        /* Residual Reduction */
        residual_reduction_blocks_kernel<<<blocks, threads, threads.x * threads.y * sizeof(double)>>>(p, rhs, flag, residual_reductions);
        residual_reduction_global_kernel<<<1, reduction_threads, reduction_threads * sizeof(double)>>>(residual_reductions, residual, blocks.x, blocks.y, p0);
        cudaDeviceSynchronize();

        // Copies residual to host code so it can be checked against eps (and printed in main vortex loop)
        cudaMemcpy(&residual_h, residual, sizeof(double), cudaMemcpyDeviceToHost);

        /* convergence? */
        if (residual_h < eps)
            break;
    }
}

void update_velocity(dim3 threads, dim3 blocks) {
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    update_velocity_u_kernel<<<blocks, threads, 0, s1>>>(u, v, p, rhs, f, g, flag);
    update_velocity_v_kernel<<<blocks, threads, 0, s2>>>(u, v, p, rhs, f, g, flag);
    cudaDeviceSynchronize();

    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);

    // Makes function inline with default stream
    cudaDeviceSynchronize();
}

void program_start(dim3 threads, dim3 blocks, int argc, char *argv[]){
    set_defaults();
    parse_args(argc, argv);

    setup();
    cudaDeviceSynchronize();

    if (verbose)
        print_opts();

    allocate_arrays();

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    setup_uvp_kernel<<<blocks, threads, 0, stream1>>>(u, v, p);
    setup_flag_kernel<<<blocks, threads, 0, stream2>>>(flag);
    cudaDeviceSynchronize();

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    // Makes function inline with default stream
    cudaDeviceSynchronize();

    boundary_conditions(threads, blocks);
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
    // Number of threads and blocks required when running one thread per grid cell
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((imax_h + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
				   (jmax_h + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);
    int reduction_threads = pow(2, ceil(log2(numBlocks.x * numBlocks.y)));

    /* Timer Initialisations */
    double total_time = get_time();

    program_start(threadsPerBlock, numBlocks, argc, argv);

    /* Main loop */
    int iters = 0;
    double t;
    for (t = 0.0; t < t_end_h; t += del_t_h, iters++)
    {
        if (!fixed_dt) {
            timestep_interval(threadsPerBlock, numBlocks, reduction_threads);
        }

        compute_tentative_velocity(threadsPerBlock, numBlocks);

        compute_rhs(threadsPerBlock, numBlocks);

        poisson(threadsPerBlock, numBlocks, reduction_threads);

        update_velocity(threadsPerBlock, numBlocks);

        boundary_conditions(threadsPerBlock, numBlocks);

        if ((iters % output_freq == 0))
        {
            printf("Step %8d, Time: %14.8e (del_t: %14.8e), Residual: %14.8e\n", iters, t + del_t_h, del_t_h, residual_h);

            if ((!no_output) && (enable_checkpoints)) {
                cudaDeviceSynchronize();
                update_host_arrays();
                write_checkpoint(iters, t + del_t_h);
            }
        }
    } /* End of main loop */
 
    cudaDeviceSynchronize();
    update_host_arrays();

    total_time = get_time() - total_time;

    printf("Step %8d, Time: %14.8e, Residual: %14.8e\n", iters, t, residual_h);
    printf("Simulation complete.\n");

    fprintf(stderr, "Timing Summary\n");
    fprintf(stderr, "Total Time: %lf\n", total_time);

    if (!no_output)
        write_result(iters, t);

    free_arrays();

    return 0;
}