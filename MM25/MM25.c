#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "MM25.h"

extern void dgemm_(char* ta,char* tb,int* m,int* n,int* k,
                  double* alpha,double* A,int* ldA,
                  double* B,int* ldB,double* beta,double* C,int* ldC);
extern void dcopy_(int* n,double* x,int* incx,double* y,int* incy);

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

void swap(double** a,double** b){
	double* t = *a;
	*a = *b;
	*b = t;
}

void mypdgemm_summa_sub(int n,double* A,double* B,double* C,double* work1,double* work2,GridInfo3D* gi){
	GridInfo gridX,gridY,gridZ;
	gridX = gi->gx;
	gridY = gi->gy;
	gridZ = gi->gz;
	int numx = gi->numx;
	int numy = gi->numy;
	int numz = gi->numz;

	int i = gridX.iam;
	int j = gridY.iam;
	int k = gridZ.iam;
	int c = numz;
	int p = gi->numglobal;
	int pc_12 = numx;
	int pc3_12 = numx / numz;

	MPI_Status status;
	int count = n*n;
	
	int incx = 1;
	int incy = 1;

	int dt = k * pc3_12;
	for(int t = dt; t < dt + pc3_12; t++){
		if(j == t){
			dcopy_(&count,A,&incx,work1,&incy);
		}
		if(i == t){
			dcopy_(&count,B,&incx,work2,&incy);
		}
		MPI_Bcast(work1,count,MPI_DOUBLE,t,gridX.comm);
		MPI_Bcast(work2,count,MPI_DOUBLE,t,gridY.comm);
		localMul(n,work1,work2,C);
	}
}

void mypdgemm_summa(int n,double* A,double* B,double* C,double* work1,double* work2,GridInfo3D* gi){
	GridInfo gridX,gridY,gridZ;
	gridZ = gi->gz;
	int count = n*n;

	MPI_Bcast(A,count,MPI_DOUBLE,0,gridZ.comm);
	MPI_Bcast(B,count,MPI_DOUBLE,0,gridZ.comm);

	mypdgemm_summa_sub(n,A,B,C,work1,work2,gi);

	int incx = 1;
	int incy = 1;
	dcopy_(&count,C,&incx,work1,&incy);
	MPI_Reduce(work1,C,count,MPI_DOUBLE,MPI_SUM,0,gridZ.comm);
}

void mypdgemm_cannon(int n,double* A,double* B,double* C,double* work1,double* work2,GridInfo3D* gi){
	GridInfo gridX,gridY,gridZ;
	gridX = gi->gx;
	gridY = gi->gy;
	gridZ = gi->gz;
	int numx = gi->numx;
	int numy = gi->numy;
	int numz = gi->numz;

	int i = gridX.iam;
	int j = gridY.iam;
	int k = gridZ.iam;
	int c = numz;
	int p = gi->numglobal;
	int pc_12 = numx;
	int pc3_12 = numx / numz;
	
	MPI_Request req[4];
	MPI_Status status[4];
	int count = n*n;


	MPI_Bcast(A,count,MPI_DOUBLE,0,gridZ.comm);
	MPI_Bcast(B,count,MPI_DOUBLE,0,gridZ.comm);
	
	int r = (j + i - k*pc3_12 + pc_12*k) % pc_12;
	int s = (j - i + k*pc3_12 + pc_12) % pc_12;

	MPI_Isend(A,count,MPI_DOUBLE,s,0,gridY.comm,&req[0]);
	MPI_Irecv(work1,count,MPI_DOUBLE,MPI_ANY_SOURCE,0,gridY.comm,&req[1]);

	int sp = (i - j + k*pc3_12 + pc_12) % pc_12;
	
	MPI_Isend(B,count,MPI_DOUBLE,sp,0,gridX.comm,&req[2]);
	MPI_Irecv(work2,count,MPI_DOUBLE,MPI_ANY_SOURCE,0,gridX.comm,&req[3]);

	MPI_Waitall(4,req,status);
	localMul(n,work1,work2,C);


	s = (j+1) % pc_12;
	sp = (i+1) % pc_12;

	for(int t = 1; t < pc3_12; t++){
		swap(&A,&work1);
		swap(&B,&work2);
		
		MPI_Isend(A,count,MPI_DOUBLE,s,t,gridY.comm,&req[0]);
		MPI_Isend(B,count,MPI_DOUBLE,sp,t,gridX.comm,&req[2]);

		r = (r - 1 + pc_12) % pc_12;

		MPI_Irecv(work1,count,MPI_DOUBLE,MPI_ANY_SOURCE,t,gridY.comm,&req[1]);
		MPI_Irecv(work2,count,MPI_DOUBLE,MPI_ANY_SOURCE,t,gridX.comm,&req[3]);
		
		MPI_Waitall(4,req,status);
		localMul(n,work1,work2,C);
	}
	
	int incx = 1;
	int incy = 1;
	dcopy_(&count,C,&incx,work1,&incy);
	MPI_Reduce(work1,C,count,MPI_DOUBLE,MPI_SUM,0,gridZ.comm);
}

void mypdgemm(int n,double* A,double* B,double* C,double* work1,double* work2,GridInfo3D* gi){
	mypdgemm_summa(n,A,B,C,work1,work2,gi);
}


int get3dComm(MPI_Comm oldComm,GridInfo3D* gi,int zdim){
	int numprocs;
	MPI_Comm_size(oldComm,&numprocs);
	
	int ndims = 3;
	int dims[3];
	int period[3] = {0,0,0};
	int reorder = 1;
	int iam;
	int coords[3];
	MPI_Comm newComm;
	
	if(zdim <= 0){
		dims[2] = 4;
	}
	else{
		dims[2] = zdim;
	}
	dims[0] = dims[1] = sqrt(numprocs/dims[2]);
	if(dims[0] * dims[1] * dims[2] != numprocs){
		fprintf(stderr,"wrong # of proc\n");
		return -1;
	}
	if(numprocs % (dims[2] * dims[2] * dims[2]) != 0){
		fprintf(stderr,"# proc mod dims[2]^3 != 0\n");
		return -1;
	}

	gi->numglobal = numprocs;
	gi->numx = dims[0];
	gi->numy = dims[1];
	gi->numz = dims[2];

	int err = MPI_Cart_create(oldComm,ndims,dims,period,reorder,&newComm);
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

	printf("%d %d %d %d\n",iam,coords[0],coords[1],coords[2]); // TODO remove
	return 0;
}


int get3dComm2(MPI_Comm oldComm,GridInfo3D* gi,int zdim){
	int numprocs;
	MPI_Comm_size(oldComm,&numprocs);
	
	int ndims = 3;
	int dims[3];
	int period[3] = {0,0,0};
	int reorder = 1;
	int iam;
	int coords[3];
	MPI_Comm newComm;
	
	if(zdim <= 0){
		dims[2] = 4;
	}
	else{
		dims[2] = zdim;
	}
	dims[0] = dims[1] = sqrt(numprocs/dims[2]);
	if(dims[0] * dims[1] * dims[2] != numprocs){
		fprintf(stderr,"wrong # of proc\n");
		return -1;
	}
	if(numprocs % (dims[2] * dims[2] * dims[2]) != 0){
		fprintf(stderr,"# proc mod dims[2]^3 != 0\n");
		return -1;
	}

	gi->numglobal = numprocs;
	gi->numx = dims[0];
	gi->numy = dims[1];
	gi->numz = dims[2];

//	int err = MPI_Cart_create(oldComm,ndims,dims,period,reorder,&newComm);
	err = MPIMY_Cart_create(oldComm,ndims,dims,period,&newComm);
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

	printf("%d %d %d %d\n",iam,coords[0],coords[1],coords[2]); // TODO remove
	return 0;
}


