#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// double rdx2 = 1.0 / (delx * delx);
//         double rdy2 = 1.0 / (dely * dely);
//         double beta_2 = -omega / (2.0 * (rdx2 + rdy2));

//         double p0 = 0.0;
//         /* Calculate sum of squares */
//         for (int i = 1; i < imax + 1; i++)
//         {
//             for (int j = 1; j < jmax + 1; j++)
//             {
//                 if (flag[i][j] & C_F)
//                 {
//                     p0 += p[i][j] * p[i][j];
//                 }
//             }
//         }

//         p0 = sqrt(p0 / fluid_cells);
//         if (p0 < 0.0001)
//         {
//             p0 = 1.0;
//         }

#define ind(i, row, row_size) ((i) + ((row) * (row_size)))

#define test_ind(i, j) ((i) * (arr_size_y) + (j))

int ROOT = 0;

int imax = 5;
int jmax = 7;

char **flag;
double **p;

/**
 * @brief Allocate a 2D array that is addressable using square brackets
 *
 * @param m The first dimension of the array
 * @param n The second dimension of the array
 * @return double** A 2D array
 */
double **alloc_2d_array(int m, int n)
{
    double **x;
    int i;

    x = (double **)malloc(m * sizeof(double *));
    x[0] = (double *)calloc(m * n, sizeof(double));
    for (i = 1; i < m; i++)
        x[i] = &x[0][i * n];
    return x;
}

char **alloc_2d_char_array(int m, int n)
{
    char **x;
    int i;

    x = (char **)malloc(m * sizeof(char *));
    x[0] = (char *)calloc(m * n, sizeof(char));
    for (i = 1; i < m; i++)
        x[i] = &x[0][i * n];
    return x;
}

int main(int argc, char **argv)
{

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    int process_num; // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &process_num);

    int rank; // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int arr_size_x = imax + 2;
    int arr_size_y = jmax + 2;

    int ROWS_PER_PROCESS = (arr_size_x - 2) / (process_num - 1);
    int ROW_REMAINDER = (arr_size_x - 2) % ROWS_PER_PROCESS;

    if (rank == ROOT)
    {
        printf("Rows to compute %d\n", arr_size_x - 2);
        printf("Number of processess including host %d\n", process_num);
        printf("Num row per process %d\n", ROWS_PER_PROCESS);
        printf("remainder %d\n", ROW_REMAINDER);

        flag = alloc_2d_char_array(arr_size_x, arr_size_y);
        p = alloc_2d_array(arr_size_x, arr_size_y);

        for (int i = 0; i < imax + 2; i++)
        {
            for (int j = 0; j < jmax + 2; j++)
            {
                flag[i][j] = 0x0010;
                p[i][j] = i;
            }
        }

        // print flag
        printf("\nflag array values\n");
        for (int i = 0; i < imax + 2; i++)
        {
            for (int j = 0; j < jmax + 2; j++)
            {
                printf("%x ", flag[i][j]);
            }
            printf("\n");
        }

        printf("\np array values\n");
        for (int i = 0; i < imax + 2; i++)
        {
            for (int j = 0; j < jmax + 2; j++)
            {
                printf("%.2f ", p[i][j]);
            }
            printf("\n");
        }
 
    }

    int* send_counts = malloc(process_num * sizeof(int));
    int* displ = malloc(process_num * sizeof(int));

    send_counts[0] = ROW_REMAINDER * arr_size_y; // Set first value to remainder elements
    displ[0] = 0;
    for (int i = 1; i < process_num; i++) {
        send_counts[i] = ROWS_PER_PROCESS * arr_size_y; // Set all other values to 1
        displ[i] = ROW_REMAINDER + ((i-1) * ROWS_PER_PROCESS * arr_size_y);
    }

    if (rank == ROOT) {
        printf("Send counts\n");
        for (int i = 0; i < process_num; i++) {
            printf("%d ", send_counts[i]);
        }

        printf("Displacement\n");
        for (int i = 0; i < process_num; i++) {
            printf("%d ", displ[i]);
        }
    }

    // Storage for p0 on each rank
    double *p0_local = malloc(sizeof(double) * send_counts[rank]);


    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatterv(p[0], send_counts, displ, MPI_DOUBLE, p0_local, p0_local[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == ROOT) {
        printf("Hello I am rank %d and I have recieved the following data\n", rank);
        for(int i = 0; i < send_counts[rank]; i++) {
            printf("%f ", p0_local[i]);
        }
    }
    MPI_Finalize();
    return 0;
}