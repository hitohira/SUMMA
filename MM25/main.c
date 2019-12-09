#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "MM25.h"	

#define SIZE_OF_MATRIX 1024 
#define TIMES 10

// proc num = 4 * x * x

int is_pow2(int n){
	while(n > 1){
		if(n % 2 != 0){
			return 0;
		}
		n /= 2;
	}
	return 1;
}

void initAll1(int n,double *a){
	for(int i = 0; i < n*n; i++){
		a[i] = 1.0;
	}
}
void initAll0(int n,double *a){
	for(int i = 0; i < n*n; i++){
		a[i] = 0.0;
	}
}
void initA(int n,double* A){
	initAll1(n,A);
}
void initB(int n,double* B){
	initAll1(n,B);
}
void initC(int n,double* C){
	initAll0(n,C);
}

int matOK(int n,double* C){
	for(int i = 0; i < n*n; i++){
		if(fabs(C[i]-n) > 1e4){
			return 0;
		}
	}
	return 1;
}


int main(int argc,char** argv){
	int numprocs,myid;
	int err;
	GridInfo3D gridinfo3d;
	GridInfo gridX,gridY,gridZ;
	double *A = NULL;
	double *B = NULL;
	double *C = NULL;
	double *work1 = NULL;
	double *work2 = NULL;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	if(!is_pow2(numprocs)){
		fprintf(stderr,"numprocs should be pow of 2\n");
		goto fine;
	}
	if(!is_pow2(SIZE_OF_MATRIX)){
		fprintf(stderr,"size of matrix should be pow of 2\n");
		goto fine;
	}
	
	for(int z = 0; z < 4; z++){
		if(z == 0){
			err = get3dComm(MPI_COMM_WORLD,&gridinfo3d,1);
		}
		else if(z == 1){
			err = get3dComm(MPI_COMM_WORLD,&gridinfo3d,4);
		}
		if(z == 2){
			err = get3dComm2(MPI_COMM_WORLD,&gridinfo3d,1);
		}
		else if(z == 3){
			err = get3dComm2(MPI_COMM_WORLD,&gridinfo3d,4);
		}
		if(err == -1){
			goto fine;
		}
		gridX = gridinfo3d.gx;
		gridY = gridinfo3d.gy;
		gridZ = gridinfo3d.gz;
		int n = SIZE_OF_MATRIX;
		int subn = n / gridinfo3d.numx;
		A = (double*)malloc(subn*subn*sizeof(double));
		B = (double*)malloc(subn*subn*sizeof(double));
		C = (double*)malloc(subn*subn*sizeof(double));
		work1 = (double*)malloc(subn*subn*sizeof(double));
		work2 = (double*)malloc(subn*subn*sizeof(double));
		if(A == NULL || B == NULL || C == NULL || work1 == NULL || work2 == NULL){
			goto fine;
		}

		double ave = 0.0;

		initA(subn,A);
		initB(subn,B);
		initC(subn,C);
		mypdgemm(subn,A,B,C,work1,work2,&gridinfo3d);
		for(int t = 0; t < TIMES; t++){
			initA(subn,A);
			initB(subn,B);
			initC(subn,C);
			MPI_Barrier(gridinfo3d.global.comm);
			double t1 = MPI_Wtime();
			mypdgemm(subn,A,B,C,work1,work2,&gridinfo3d);
			MPI_Barrier(gridinfo3d.global.comm);
			double t2 = MPI_Wtime();
			ave += (t2 - t1);
		}
		ave /= TIMES;
		if(myid == 0){
			printf("%s : ",z == 0 ? "2D" : "3D");
			printf("ave. = %f\n",ave);
			if(!matOK(subn,C)){
				printf("wrong pgemm\n");
			}
		}
	}

fine:
	if(A != NULL) free(A);
	if(B != NULL) free(B);
	if(C != NULL) free(C);
	if(work1 != NULL) free(work1);
	if(work2 != NULL) free(work2);
	MPI_Finalize();
	return 0;
}


