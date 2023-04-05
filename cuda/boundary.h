#ifndef BOUNDARY_H
#define BOUNDARY_H

__global__ void apply_boundary_conditions(double* u, double* v, double* p, double* rhs, double* f, double* g, char* flag);

#endif