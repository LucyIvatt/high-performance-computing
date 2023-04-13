#include "data.h"
#include <stdio.h>

#define ind(i, j, m) ((i) * (m) + (j))

__constant__ double tau = 0.5;   /* Safety factor for timestep control */
__constant__ double omega = 1.7; /* Relaxation parameter for SOR */
__constant__ double y = 0.9;     /* Gamma, Upwind differencing factor in PDE discretisation */

__constant__ double Re = 500.0; /* Reynolds number */
__constant__ double ui = 1.0;   /* Initial X velocity */
__constant__ double vi = 0.0;   /* Initial Y velocity */

/* CONSTANTS NEEDED ON BOTH GPU AND HOST DUE TO PARSED ARGUMENTS */
__constant__ int u_size_x, u_size_y;
__constant__ int v_size_x, v_size_y;
__constant__ int p_size_x, p_size_y;
__constant__ int flag_size_x, flag_size_y;
__constant__ int g_size_x, g_size_y;
__constant__ int f_size_x, f_size_y;
__constant__ int rhs_size_x, rhs_size_y;

__constant__ double rdx2, rdy2, beta_2;

__constant__ int imax;     /* Number of cells horizontally */
__constant__ int jmax;     /* Number of cells vertically */
__constant__ double t_end; /* Simulation runtime */
__constant__ double delx;
__constant__ double dely;

__constant__ double mx;
__constant__ double my;
__constant__ double rad1;

__device__ int fluid_cells = 0;
__device__ double del_t; /* Duration of each timestep */

__global__ void setup_uvp_kernel(double *u, double *v, double *p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < imax + 2 && j < jmax + 2)
    {
        u[ind(i, j, u_size_y)] = ui;
        v[ind(i, j, v_size_y)] = vi;
        p[ind(i, j, p_size_y)] = 0.0;
    }
}

__global__ void setup_flag_kernel(char *flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    /* Mark a circular obstacle as boundary cells, the rest as fluid */
    if (i > 0 && i <= imax && j > 0 && j <= jmax)
    {
        double x = (i - 0.5) * delx - mx;
        double y = (j - 0.5) * dely - my;
        flag[ind(i, j, flag_size_y)] = (x * x + y * y <= rad1 * rad1) ? C_B : C_F;
    }

    /* Mark the north & south boundary cells */
    if (i <= imax + 1)
    {
        flag[ind(i, 0, flag_size_y)] = C_B;
        flag[ind(i, jmax + 1, flag_size_y)] = C_B;
    }
    /* Mark the east and west boundary cells */
    if (j > 0 && j <= jmax)
    {
        flag[ind(0, j, flag_size_y)] = C_B;
        flag[ind(imax + 1, j, flag_size_y)] = C_B;
    }

    /* flags for boundary cells */
    if (i > 0 && i <= imax && j > 0 && j <= jmax)
    {
        if (!(flag[ind(i, j, flag_size_y)] & C_F))
        {
            fluid_cells--;
            if (flag[ind(i - 1, j, flag_size_y)] & C_F)
                flag[ind(i, j, flag_size_y)] |= B_W;
            if (flag[ind(i + 1, j, flag_size_y)] & C_F)
                flag[ind(i, j, flag_size_y)] |= B_E;
            if (flag[ind(i, j - 1, flag_size_y)] & C_F)
                flag[ind(i, j, flag_size_y)] |= B_S;
            if (flag[ind(i, j + 1, flag_size_y)] & C_F)
                flag[ind(i, j, flag_size_y)] |= B_N;
        }
    }
}

__global__ void boundary_conditions_WE_kernel(double* u, double* v){
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < jmax + 2)
    {
        /* Fluid freely flows in from the west */
        u[ind(0, j, u_size_y)] = u[ind(1, j, u_size_y)];
        v[ind(0, j, v_size_y)] = v[ind(1, j, v_size_y)];

        /* Fluid freely flows out to the east */
        u[ind(imax, j, u_size_y)] = u[ind(imax - 1, j, u_size_y)];
        v[ind(imax + 1, j, v_size_y)] = v[ind(imax, j, v_size_y)];
    }
}

__global__ void boundary_conditions_NS_kernel(double* u, double* v){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < imax + 2)
    {
        /* The vertical velocity approaches 0 at the north and south
         * boundaries, but fluid flows freely in the horizontal direction */
        v[ind(i, jmax, v_size_y)] = 0.0;
        u[ind(i, jmax + 1, u_size_y)] = u[ind(i, jmax, u_size_y)];

        v[ind(i, 0, v_size_y)] = 0.0;
        u[ind(i, 0, u_size_y)] = u[ind(i, 1, u_size_y)];
    }
}

/**
 * @brief Given the boundary conditions defined by the flag matrix, update
 * the u and v velocities. Also enforce the boundary conditions at the
 * edges of the matrix.
 */
__global__ void boundary_conditions_noslip_kernel(double *u, double *v, char *flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;


    /* Apply no-slip boundary conditions to cells that are adjacent to
     * internal obstacle cells. This forces the u and v velocity to
     * tend towards zero in these cells.
     */

    if (i > 0 && i < imax + 1 && j > 0 && j < jmax + 1)
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

__global__ void apply_boundary_conditions_west_edge_kernel(double *u, double *v)
{
    /* Finally, fix the horizontal velocity at the  western edge to have
     * a continual flow of fluid into the simulation.
     */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i == 0 && j == 0)
        v[ind(0, 0, v_size_y)] = 2 * vi - v[ind(1, 0, v_size_y)];

    if (j > 0 && j < jmax+1)
    {
        u[ind(0, j, u_size_y)] = ui;
        v[ind(0, j, v_size_y)] = 2 * vi - v[ind(1, j, v_size_y)];
    }
}

__global__ void tentative_velocity_update_f_kernel(double *u, double *v, double *f, char *flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < imax && j > 0 && j < jmax + 1)
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

__global__ void tentative_velocity_update_g_kernel(double *u, double *v, double *g, char *flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < imax + 1 && j > 0 && j < jmax)
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

__global__ void tentative_velocity_f_boundaries_kernel(double *f, double *u)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j > 0 && j < jmax + 1)
    {
        /* f & g at external boundaries */
        f[ind(0, j, f_size_y)] = u[ind(0, j, u_size_y)];
        f[ind(imax, j, f_size_y)] = u[ind(imax, j, u_size_y)];
    }
}


__global__ void tentative_velocity_g_boundaries_kernel(double *g, double *v){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < imax + 1)
    {
        g[ind(i, 0, g_size_y)] = v[ind(i, 0, v_size_y)];
        g[ind(i, jmax, g_size_y)] = v[ind(i, jmax, v_size_y)];
    }
}


/**
 * @brief Calculate the right hand side of the pressure equation
 *
 */
__global__ void compute_rhs_kernel(double *u, double *v, double *p, double *rhs, double *f, double *g, char *flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < imax + 1 && j > 0 && j < jmax + 1)
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

__global__ void p0_reduction_blocks_kernel(double *p, char *flag, double *global_reductions)
{
    extern __shared__ double block_reductions[];

    int i = ind(blockIdx.x, threadIdx.x, blockDim.x);
    int j = ind(blockIdx.y, threadIdx.y, blockDim.y);

    int array_ind = ind(i, j, jmax + 2);
    int b_tid = ind(threadIdx.x, threadIdx.y, blockDim.y); // Block Thread ID

    int t_per_b = blockDim.x * blockDim.y;            // Threads per block (rounded to nearest even number)
    int bid = ind(blockIdx.x, blockIdx.y, gridDim.y); // Block id (within grid)

    // If the thread is valid -->
    if (i > 0 && i < imax + 1 && j > 0 && j < jmax + 1)
    {
        // copied the value from the data array into the blocks shared memory
        if (flag[ind(i, j, jmax + 2)] & C_F)
        {
            block_reductions[b_tid] = p[array_ind] * p[array_ind];
        }
    }
    else
    {
        // otherwise sets the shared memory value to 0 (so that this will never be picked if compared)
        block_reductions[b_tid] = 0;
    }
    __syncthreads();

    // Uses sequental addressing for a reduction
    for (unsigned int s = t_per_b / 2; s > 0; s /= 2)
    {
        if (b_tid < s)
        {
            block_reductions[b_tid] = block_reductions[b_tid] + block_reductions[b_tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (b_tid == 0)
        global_reductions[bid] = block_reductions[0];
}

__global__ void p0_reduction_global_kernel(double *global_reductions, double *p0, int num_blocks_x, int num_blocks_y)
{
    extern __shared__ double final_reductions[];

    int b_tid = ind(threadIdx.x, threadIdx.y, blockDim.y);

    int t_per_b = blockDim.x * blockDim.y;

    if (b_tid < num_blocks_x * num_blocks_y)
    {
        final_reductions[b_tid] = global_reductions[b_tid];
    }
    else
    {
        final_reductions[b_tid] = 0;
    }
    __syncthreads();

    // Uses sequental addressing for a reduction
    for (unsigned int s = t_per_b / 2; s > 0; s /= 2)
    {
        if (b_tid < s)
        {
            final_reductions[b_tid] = final_reductions[b_tid] + final_reductions[b_tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (b_tid == 0)
    {
        *p0 = final_reductions[0];

        *p0 = sqrt(*p0 / fluid_cells);
        if (*p0 < 0.0001)
        {
            *p0 = 1.0;
        }
    }
}

__global__ void star_computation_kernel(double *u, double *v, double *p, double *rhs, double *f, double *g, char *flag, int rb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < imax + 1 && j > 0 && j < jmax + 1 && (i + j) % 2 == rb)
    {

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

__global__ void residual_reduction_blocks_kernel(double *p, double *rhs, char *flag, double *global_reductions)
{
    extern __shared__ double block_reductions[];

    int i = ind(blockIdx.x, threadIdx.x, blockDim.x);
    int j = ind(blockIdx.y, threadIdx.y, blockDim.y);

    int b_tid = ind(threadIdx.x, threadIdx.y, blockDim.y); // Block Thread ID

    int t_per_b = blockDim.x * blockDim.y;            // Threads per block (rounded to nearest even number)
    int bid = ind(blockIdx.x, blockIdx.y, gridDim.y); // Block id (within grid)

    // If the thread is valid -->
    if (i > 0 && i < imax + 1 && j > 0 && j < jmax + 1 && (flag[ind(i, j, flag_size_y)] & C_F))
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
        block_reductions[b_tid] = add * add;
    }
    else
    {
        // otherwise sets the shared memory value to 0 (so that this will never be picked if compared)
        block_reductions[b_tid] = 0;
    }
    __syncthreads();

    // Uses sequental addressing for a reduction
    for (unsigned int s = t_per_b / 2; s > 0; s /= 2)
    {
        if (b_tid < s)
        {
            block_reductions[b_tid] = block_reductions[b_tid] + block_reductions[b_tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (b_tid == 0)
        global_reductions[bid] = block_reductions[0];
}

__global__ void residual_reduction_global_kernel(double *global_reductions, double *residual, int num_blocks_x, int num_blocks_y, double *p0)
{
    extern __shared__ double final_reductions[];

    int b_tid = ind(threadIdx.x, threadIdx.y, blockDim.y);

    int t_per_b = blockDim.x * blockDim.y;

    if (b_tid < num_blocks_x * num_blocks_y)
    {
        final_reductions[b_tid] = global_reductions[b_tid];
    }
    else
    {
        final_reductions[b_tid] = 0;
    }
    __syncthreads();

    // Uses sequental addressing for a reduction
    for (unsigned int s = t_per_b / 2; s > 0; s /= 2)
    {
        if (b_tid < s)
        {
            final_reductions[b_tid] = final_reductions[b_tid] + final_reductions[b_tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (b_tid == 0)
    {
        *residual = final_reductions[0];
        *residual = sqrt((double)*residual / fluid_cells) / *p0;
    }
}

__global__ void update_velocity_u_kernel(double *u, double *v, double *p, double *rhs, double *f, double *g, char *flag){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < imax - 2 && j > 0 && j < jmax - 1)
    {
        /* only if both adjacent cells are fluid cells */
        if ((flag[ind(i, j, flag_size_y)] & C_F) && (flag[ind(i + 1, j, flag_size_y)] & C_F))
        {
            u[ind(i, j, u_size_y)] = f[ind(i, j, f_size_y)] - (p[ind(i + 1, j, p_size_y)] - p[ind(i, j, p_size_y)]) * del_t / delx;
        }
    }
}

/**
 * @brief Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
__global__ void update_velocity_v_kernel(double *u, double *v, double *p, double *rhs, double *f, double *g, char *flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < imax - 1 && j > 0 && j < jmax - 2)
    {
        /* only if both adjacent cells are fluid cells */
        if ((flag[ind(i, j, flag_size_y)] & C_F) && (flag[ind(i, j + 1, flag_size_y)] & C_F))
        {
            v[ind(i, j, v_size_y)] = g[ind(i, j, g_size_y)] - (p[ind(i, j + 1, p_size_y)] - p[ind(i, j, p_size_y)]) * del_t / dely;
        }
    }
}

/**
 * @brief Set the timestep size so that we satisfy the Courant-Friedrichs-Lewy
 * conditions. Otherwise the simulation becomes unstable.
 */
__global__ void set_timestep_interval_kernel(double *umax_g, double *vmax_g)
{
    *umax_g = fmax(*umax_g, 1.0e-10);
    *vmax_g = fmax(*vmax_g, 1.0e-10);

    double deltu = delx / *umax_g;
    double deltv = dely / *vmax_g;
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

__global__ void abs_max_reduction_blocks_kernel(double *array, double *global_reductions, int version)
{
    extern __shared__ double block_reductions[];

    int i = ind(blockIdx.x, threadIdx.x, blockDim.x);
    int j = ind(blockIdx.y, threadIdx.y, blockDim.y);

    int array_ind = ind(i, j, jmax + 2);
    int b_tid = ind(threadIdx.x, threadIdx.y, blockDim.y); // Block Thread ID

    int t_per_b = blockDim.x * blockDim.y;            // Threads per block (rounded to nearest even number)
    int bid = ind(blockIdx.x, blockIdx.y, gridDim.y); // Block id (within grid)

    // If the thread is valid -->
    if ((version == 0 && i < imax + 2 && j > 0 && j < jmax + 2) || (version == 1 && i > 1 && i < imax + 2 && j < jmax + 2))
    {
        block_reductions[b_tid] = fabs((double)array[array_ind]);
    }
    else
    {
        // otherwise sets the shared memory value to 0 (so that this will never be picked if compared)
        block_reductions[b_tid] = 0;
    }
    __syncthreads();

    // Uses sequental addressing for a reduction
    for (unsigned int s = t_per_b / 2; s > 0; s /= 2)
    {
        if (b_tid < s)
        {
            block_reductions[b_tid] = fmax((double)block_reductions[b_tid], block_reductions[b_tid + s]);
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (b_tid == 0)
        global_reductions[bid] = block_reductions[0];
}

__global__ void abs_max_reduction_global_kernel(double *global_reductions, double *output_val, int num_blocks_x, int num_blocks_y)
{
    extern __shared__ double final_reductions[];

    int b_tid = ind(threadIdx.x, threadIdx.y, blockDim.y);

    int t_per_b = blockDim.x * blockDim.y;

    if (b_tid < num_blocks_x * num_blocks_y)
    {
        final_reductions[b_tid] = global_reductions[b_tid];
    }
    else
    {
        final_reductions[b_tid] = 0;
    }
    __syncthreads();

    // Uses sequental addressing for a reduction
    for (unsigned int s = t_per_b / 2; s > 0; s /= 2)
    {
        if (b_tid < s)
        {
            final_reductions[b_tid] = fmax((double)final_reductions[b_tid], final_reductions[b_tid + s]);
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (b_tid == 0)
    {
        *output_val = final_reductions[0];
    }
}