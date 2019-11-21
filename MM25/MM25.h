#ifndef MM25_H
#define MM25_H

#include <mpi.h>

typedef struct{
	int iam;
	MPI_Comm comm;
} GridInfo;
typedef struct{
	int numx,numy,numz;
	GridInfo global,gx,gy,gz;
} GridInfo3D;

void mypdgemm(int n,double* A.double* B,double* C,GridInfo3D* gi);
int get3dComm(MPI_Comm oldComm,GridInfo3D* gi);

#endif