#ifndef DATA_H
#define DATA_H

#define C_B 0x0000 /* This cell is an obstacle/boundary cell */
#define B_N 0x0001 /* This obstacle cell has a fluid cell to the north */
#define B_S 0x0002 /* This obstacle cell has a fluid cell to the south */
#define B_W 0x0004 /* This obstacle cell has a fluid cell to the west */
#define B_E 0x0008 /* This obstacle cell has a fluid cell to the east */
#define B_NW (B_N | B_W)
#define B_SW (B_S | B_W)
#define B_NE (B_N | B_E)
#define B_SE (B_S | B_E)
#define B_NSEW (B_N | B_S | B_E | B_W)

#define C_F 0x0010 /* This cell is a fluid cell */

#define ROOT 0
#define ind(i, j) ((i) * (arr_size_y) + (j))
#define ind_ret(i, j) (((i)+1) * (arr_size_y) + (j))

extern double xlength; /* Width of simulated domain */
extern double ylength; /* Height of simulated domain */
extern int imax;       /* Number of cells horizontally */
extern int jmax;       /* Number of cells vertically */

extern double t_end; /* Simulation runtime */
extern double del_t; /* Duration of each timestep */
extern double tau;   /* Safety factor for timestep control */

extern int itermax;  /* Maximum number of iterations in SOR */
extern double eps;   /* Stopping error threshold for SOR */
extern double omega; /* Relaxation parameter for SOR */
extern double y;     /* Gamma, Upwind differencing factor in PDE */

extern double Re; /* Reynolds number */
extern double ui; /* Initial X velocity */
extern double vi; /* Initial Y velocity */

extern double rdx2;
extern double rdy2;
extern double beta_2;

extern int fluid_cells;

extern double delx, dely;

// Grids used for veclocities, pressure, rhs, flag and temporary f and g arrays

extern int arr_size_x, arr_size_y;

extern double **u;
extern double **v;
extern double **p;
extern double **rhs;
extern double **f;
extern double **g;
extern char **flag;

double **alloc_2d_array(int m, int n);
char **alloc_2d_char_array(int m, int n);
void free_2d_array(void **array);

#endif