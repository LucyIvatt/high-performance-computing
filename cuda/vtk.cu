#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include "vtk.h"
#include "data.h"

char checkpoint_basename[1024];
char result_filename[1024];

/**
 * @brief Set the default basename for file output to out/vortex
 *
 */
void set_default_base()
{
    set_basename("cuda/cuda_vortex");
}

/**
 * @brief Set the basename for file output
 *
 * @param base Basename string
 */
void set_basename(char *base)
{
    checkpoint_basename[0] = '\0';
    result_filename[0] = '\0';
    sprintf(checkpoint_basename, "%s-%%d.vtk", base);
    sprintf(result_filename, "%s.vtk", base);
}

/**
 * @brief Get the basename for file output
 *
 * @return char* Basename string
 */
char *get_basename()
{
    return checkpoint_basename;
}

/**
 * @brief Write a checkpoint VTK file (with the iteration number in the filename)
 *
 * @param iteration The current iteration number
 * @return int Return whether the write was successful
 */
int write_checkpoint(int iters, double t)
{
    char filename[1024];
    sprintf(filename, checkpoint_basename, iters);
    return write_vtk(filename, iters, t);
}

/**
 * @brief Write the final output to a VTK file
 *
 * @return int Return whether the write was successful
 */
int write_result(int iters, double t)
{
    return write_vtk(result_filename, iters, t);
}

/**
 * @brief Write a VTK file with the current state of the simulation
 *
 * @param filename The filename to write out
 * @return int Return whether the write was successful
 */
int write_vtk(char *filename, int iters, double t)
{
    FILE *f = fopen(filename, "w");
    if (f == NULL)
    {
        perror("Error");
        return -1;
    }

    // Write the VTK header information
    fprintf(f, "# vtk DataFile Version 3.0\n");
    fprintf(f, "Vortex Output\n");
    fprintf(f, "ASCII\n");
    fprintf(f, "DATASET STRUCTURED_POINTS\n");

    // Write out data for the simulation time and step number
    fprintf(f, "FIELD FieldData 2\n");
    fprintf(f, "TIME 1 1 double\n");
    fprintf(f, "%lf\n", t);
    fprintf(f, "CYCLE 1 1 int\n");
    fprintf(f, "%d\n", iters);

    // Write out the dimensions of the grid
    fprintf(f, "DIMENSIONS %d %d 1\n", arr_size_x_h, arr_size_y_h);
    fprintf(f, "ORIGIN 0 0 0\n");
    fprintf(f, "SPACING 1 1 1\n");

    // Write out the u variable
    int points = arr_size_x_h * arr_size_y_h;
    fprintf(f, "POINT_DATA %d\n", points);
    fprintf(f, "SCALARS u double 1\n");
    fprintf(f, "LOOKUP_TABLE default\n");

    for (int j = 0; j < arr_size_y_h; j++)
    {
        for (int i = 0; i < arr_size_x_h; i++)
            fprintf(f, "%.12e ", u_host[ind_h(i, j)]);
        fprintf(f, "\n");
    }

    // Write out the v variable
    fprintf(f, "\nSCALARS v double 1\n");
    fprintf(f, "LOOKUP_TABLE default\n");

    for (int j = 0; j < arr_size_y_h; j++)
    {
        for (int i = 0; i < arr_size_x_h; i++)
            fprintf(f, "%.12e ", v_host[ind_h(i, j)]);
        fprintf(f, "\n");
    }

    // Write out the p variable
    fprintf(f, "\nSCALARS p double 1\n");
    fprintf(f, "LOOKUP_TABLE default\n");

    for (int j = 0; j < arr_size_y_h; j++)
    {
        for (int i = 0; i < arr_size_x_h; i++)
            fprintf(f, "%.12e ", p_host[ind_h(i, j)]);
        fprintf(f, "\n");
    }

    // Write out the flag variable
    fprintf(f, "\nSCALARS flag int 1\n");
    fprintf(f, "LOOKUP_TABLE default\n");

    for (int j = 0; j < arr_size_y_h; j++)
    {
        for (int i = 0; i < arr_size_x_h; i++)
            fprintf(f, "%d ", flag_host[ind_h(i, j)]);
        fprintf(f, "\n");
    }

    fclose(f);
    return 0;
}