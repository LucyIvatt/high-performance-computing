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

#define INDEX_2D(i, j, m) ((i*m+j))

#define C_F 0x0010 /* This cell is a fluid cell */

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

extern int fluid_cells;

extern double delx, dely;

// Grids used for veclocities, pressure, rhs, flag and temporary f and g arrays
struct DoubleArray2D {
    double* data;
    int size_x;
    int size_y;
};

struct CharArray2D {
    char* data;
    int size_x;
    int size_y;
};

extern int u_size_x, u_size_y;
extern double *u_h;
extern struct DoubleArray2D u_h_array;

extern int v_size_x, v_size_y;
extern double *v_h;
extern struct DoubleArray2D v_h_array;

extern int p_size_x, p_size_y;
extern double *p_h;
extern struct DoubleArray2D p_h_array;

extern int rhs_size_x, rhs_size_y;
extern double *rhs_h;
extern struct DoubleArray2D rhs_h_array;

extern int f_size_x, f_size_y;
extern double *f_h;
extern struct DoubleArray2D f_h_array;

extern int g_size_x, g_size_y;
extern double *g_h;
extern struct DoubleArray2D g_h_array;

extern int flag_size_x, flag_size_y;
extern char *flag_h;
extern struct CharArray2D flag_h_array;


double *alloc_2d_array(int m, int n);
char *alloc_2d_char_array(int m, int n);
char *copy_char_array_to_device(int m, int n, char *src);
double *copy_double_array_to_device(int m, int n, double *src);
void free_2d_array_device(void **array);
void free_2d_array_host(void **array);

#endif