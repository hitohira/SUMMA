#ifndef MM25_H
#define MM25_H

#include <mpi.h>

extern int MPIMY_Cart_create(MPI_Comm comm_old,int ndims,int dims[],int periods[],MPI_Comm* comm_cart);
extern int MPIMY_Cart_create2(MPI_Comm comm_old,int ndims,int dims[],int periods[],MPI_Comm* comm_cart);
extern int MPIMY_Cart_create_extra(MPI_Comm comm_old,int ndims,int dims[],int periods[],MPI_Comm* comm_cart);

void initGlobCommTime();
double getGlobCommTime();

typedef struct{
	int iam;
	MPI_Comm comm;
} GridInfo;
typedef struct{
	int numglobal,numx,numy,numz;
	GridInfo global,gx,gy,gz;
} GridInfo3D;

void localMul(int size,double* A,double* B,double* C);
void mypdgemm(int n,double* A,double* B,double* C,double* work1,double* work2,GridInfo3D* gi);
void para_bcast(int n,double* A,double* B,double* C,double* work1,double* work2,GridInfo3D* gi);

// if zdim <= 0, dims_z = 4
int get3dComm(MPI_Comm oldComm,GridInfo3D* gi,int zdim);
int get3dComm2(MPI_Comm oldComm,GridInfo3D* gi,int zdim);
int get3dComm3(MPI_Comm oldComm,GridInfo3D* gi,int zdim);
int get3dComm4(MPI_Comm oldComm,GridInfo3D* gi,int zdim);

#endif
