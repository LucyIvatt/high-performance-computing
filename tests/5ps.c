#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define read_ind(i, j) ((i) * (arr_size_y) + (j))

#define write_ind(i, j) (((i) - 1) * (arr_size_y) + (j))

int ROOT = 0;

int imax = 5;
int jmax = 6;

char **flag;
double **p, **rhs;

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
        rhs = alloc_2d_array(arr_size_x, arr_size_y);
        p = alloc_2d_array(arr_size_x, arr_size_y);

        for (int i = 0; i < imax + 2; i++)
        {
            for (int j = 0; j < jmax + 2; j++)
            {
                flag[i][j] = (i + j) % 2 == 0 ? 0x0001 : 0x0004;
                p[i][j] = i;
                rhs[i][j] = -1;
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

        printf("\nrhs array values\n");
        for (int i = 0; i < imax + 2; i++)
        {
            for (int j = 0; j < jmax + 2; j++)
            {
                printf("%.2f ", rhs[i][j]);
            }
            printf("\n");
        }
    }
    for (int rb = 0; rb < 1; rb++)
    {
        if (rank == ROOT)
        {
            // Sends the rows to the processes
            for (int r = 1; r < process_num; r++)
            {
                printf("%d\n", (ROWS_PER_PROCESS * (r - 1)));
                printf("%d\n", (arr_size_y * (ROWS_PER_PROCESS + 2)));

                int start_loc = (ROWS_PER_PROCESS * (r - 1));
                int send_size = arr_size_y * (ROWS_PER_PROCESS + 2);

                MPI_Send(p[start_loc], send_size, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
                MPI_Send(rhs[start_loc], send_size, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
                MPI_Send(flag[start_loc], send_size, MPI_CHAR, r, 0, MPI_COMM_WORLD);
            }
            // Recives the new row and updates u
            for (int r = 1; r < process_num; r++)
            {
                int start_loc = 1 + (ROWS_PER_PROCESS * (r - 1));
                int recv_size = arr_size_y * ROWS_PER_PROCESS;
                MPI_Recv(p[start_loc], recv_size, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            printf("\nnew p array values\n");
            for (int i = 0; i < imax + 2; i++)
            {
                for (int j = 0; j < jmax + 2; j++)
                {
                    printf("%.2f ", p[i][j]);
                }
                printf("\n");
            }
        }

        else if (rank > 0)
        {
            int recv_size = arr_size_y * (ROWS_PER_PROCESS + 2);

            double p_buff[recv_size];
            double rhs_buff[recv_size];
            char flag_buff[recv_size];

            MPI_Recv(p_buff, recv_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(rhs_buff, recv_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(flag_buff, recv_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // printf("RECIEVED RHS\n");
            // for (int i = 0; i < ROWS_PER_PROCESS + 2; i++)
            // {
            //     for (int j = 0; j < arr_size_y; j++)
            //     {
            //         printf("%.2f ", rhs_buff[read_ind(i, j)]);
            //     }
            //     printf("\n");
            // }

            // printf("RECIEVED FLAG\n");
            // for (int i = 0; i < ROWS_PER_PROCESS + 2; i++)
            // {
            //     for (int j = 0; j < arr_size_y; j++)
            //     {
            //         printf("%x ", flag_buff[read_ind(i, j)]);
            //     }
            //     printf("\n");
            // }

            int send_size = arr_size_y * ROWS_PER_PROCESS;
            double p_send[arr_size_y * ROWS_PER_PROCESS];

            for (int i = 1; i < ROWS_PER_PROCESS + 1; i++)
            {
                for (int j = 1; j < arr_size_y - 1; j++)
                {
                    if ((i + j) % 2 != rb)
                    {
                        p_send[(write_ind(i, j))] = p_buff[(read_ind(i, j))] * p_buff[(read_ind(i, j))];
                    }
                    else
                    {
                        p_send[(write_ind(i, j))] = p_buff[(read_ind(i, j))];
                    }
                }
                // Sets values at the end of the rows to what they already were
                p_send[write_ind(i, 0)] = p_buff[read_ind(i, 0)];
                p_send[write_ind(i, arr_size_y - 1)] = p_buff[read_ind(i, arr_size_y - 1)];
            }

            MPI_Send(&(p_send[0]), send_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    //     // Recives the new row and updates u
    //     for (int r = 1; r < process_num; r++)
    //     {
    //         int start_loc = 1 + (ROWS_PER_PROCESS * (r - 1));
    //         int recv_size = arr_size_y * ROWS_PER_PROCESS;
    //         MPI_Recv(rhs[start_loc], recv_size, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //     }

    //     if (ROW_REMAINDER > 0)
    //     {
    //         int start_loc = 1 + (ROWS_PER_PROCESS * (process_num-1));
    //         printf("remainder start %d\n", start_loc);

    //         for (int i = start_loc; i < imax + 1; i++)
    //         {
    //             for (int j = 1; j < jmax + 1; j++)
    //             {
    //                 rhs[i][j] = f[i][j] * g[i][j];
    //             }
    //         }
    //     }

    //     // print u
    //     printf("\nrhs array values\n");
    //     for (int i = 0; i < imax + 2; i++)
    //     {
    //         for (int j = 0; j < jmax + 2; j++)
    //         {
    //             printf("%.2f ", rhs[i][j]);
    //         }
    //         printf("\n");
    //     }
    // }

    //     int send_size = arr_size_y * ROWS_PER_PROCESS;
    //     double rhs_buff_send[arr_size_y * ROWS_PER_PROCESS];

    //     // Calculates the square of the numbers
    //     for (int i = 0; i < ROWS_PER_PROCESS; i++)
    //     {
    //         for (int j = 1; j < arr_size_y - 1; j++)
    //         {
    //             if (flag_buff[test_ind(i, j)] & 0x0010)
    //             {
    //                 rhs_buff_send[test_ind(i, j)] = f_buff[test_ind(i + 1, j)] * g_buff[test_ind(i + 1, j)];
    //             }
    //         }

    //         // Sets values at the end of the rows to what they already were
    //         rhs_buff_send[test_ind(i, 0)] = rhs_buff_recv[test_ind(i + 1, 0)];
    //         rhs_buff_send[test_ind(i, arr_size_y - 1)] = rhs_buff_recv[test_ind(i + 1, arr_size_y - 1)];
    //     }

    //     MPI_Send(&(rhs_buff_send[0]), send_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    // }

    MPI_Finalize();
    return 0;
}