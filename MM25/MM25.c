#include <mpi.c>
#include <stdio.h>
#include <stdlib.h>

#include "MM25.h"

extern void dgemm_(char* ta,char* tb,int* m,int* n,int* k,
                  double* alpha,double* A,int* ldA,
                  double* B,int* ldB,double* beta,double* C,int* ldC);


//////////////////////////////////////////////////////////////////	
/// functions
//////////////////////////////////////////////////////////////////	

void localMul(int size,double* A,double* B,double* C){
	int m,n,k,ldA,ldB,ldC;
	m = n = k = ldA = ldB = ldC = size;
	double alpha,beta;
	alpha = beta = 1.0;

	dgemm_("N","N",&m,&n,&k,&alpha,A,&ldA,B,&ldB,&beta,C,&ldC);
}

void mypdgemm(int n,double* A,double* B,double* C,GridInfo3D* gi){
	// TODO
}


int get3dComm(MPI_Comm oldComm,GridInfo3D* gi){
	
	int ndims = 3;
	int dims[3];
	int preriod[3] = {0,0,0};
	int reorder = 1;
	int iam;
	int coords[3];
	MPI_Comm newComm;
	
	dims[2] = 4;
	dims[0] = dims[1] = sqrt(numprocs/4);
	if(dims[0] * dims[1] * dims[2] != numprocs){
		fprintf(stderr,"wrong # of proc\n");
		return -1;
	}

	gi->numx = dims[0];
	gi->numy = dims[1];
	gi->numz = dims[2];

	int err = MPI_Cart_create(oldComm,ndim,dims,period,reorder,&newComm);
	if(err != MPI_SUCCESS){
		fprintf(stderr,"cart create error\n");
		return -1;
	}
	gi->global.comm = newComm;
	MPI_Comm_rank(newComm,&iam);
	gi->global.iam = iam;
	MPI_Cart_coords(newComm,iam,ndims,coords);
	
	int xc[3] = {1,0,0};
	int yc[3] = {0,1,0};
	int zc[3] = {0,0,1};

	MPI_Cart_sub(newComm,xc,&(gi->gx.comm));
	MPI_Cart_sub(newComm,yc,&(gi->gy.comm));
	MPI_Cart_sub(newComm,zc,&(gi->gz.comm));

	gi->gx.iam = coords[0];
	gi->gy.iam = coords[1];
	gi->gz.iam = coords[2];
	return 0;
}


