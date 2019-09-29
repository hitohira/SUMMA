#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>

// 0  1  2  3
// 4  5  6  7
// 8  9 10 11
//12 13 14 15 

// each val is positive and power of 2, N >= Height * K, N >= Width * K
// Height and Width depend on Num of MPI proc
#define N 1024
#define Height 4
#define Width 4
#define K 2

int nw,nh;
double *A,*B,*C;
double *work1,*work2;

void initMat(double* A){
	int i;
	#pragma omp parallel for
	for(i = 0; i < nh * nw; i++){
		A[i] = 1.0;
		B[i] = 1.0;
		C[i] = 0.0;
	}
}

void arrayCopy(double* src,int num,double* dst){
	// copy
	int i;
	#pragma omp parallel for
	for(i = 0; i < num; i++){
		dst[i] = src[i];
	}
}

void broadcast(MPI_Comm comm,int root,int id,double* src,int num,double* dst){
	if(id == root){
		arrayCopy(src,num,dst);
	}
	MPI_Bcast(dst,num,MPI_DOUBLE,root,comm);
}

// A = (n*k)^T, B = k*m, C = n*m
void myDGEMM(int n,int m,int k,double* A,double* B,double* C){
	int p,q,r;
	#pragma omp parallel
	{
		for(p = 0; p < k; p++){
			for(q = 0; q < n; q++){
				#pragma omp for
				for(r = 0; r < m; r++){
					C[q*n+r] += A[p*n+q] * B[p*m+r];
				}
			}
		}
	}
}

bool myDGEMMok(){
	double a[16];
	double b[16];
	double c[16],d[16];
	for(int i = 0; i < 16; i++){
		a[i] = 1.0 * rand() / RAND_MAX;
		b[i] = 1.0 * rand() / RAND_MAX;
		c[i] = 0.0;
		d[i] = 0.0;
	}
	myDGEMM(4,4,4,a,b,d);
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			for(int k = 0; k < 4; k++){
				c[i*4+j] += A[i*4+k] * B[k*4+j];
			}
		}
	}
	double eps = 1e-12;
	for(int i = 0; i < 16; i++){
		if(fabs(c[i]-d[i]) > eps){
			return false;
		}
	}
	return true;
}


int main(int argc,char** argv){
	int myid,numproc;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numproc);
	MPI_Comm_Rank(MPI_COMM_WORLD,&myid);

	// test local MatMatMul
	if(!myDGEMMok() && myid == 0){
		printf("local MMM wrong\n");
		return 0;
	}
	
	MPI_Comm comm_row,comm_col;

	MPI_Comm_split(MPI_COMM_WORLD,myid % Width, myid / Width,&comm_row);
	MPI_Comm_split(MPI_COMM_WORLD,myid / Width, myid % Width,&comm_col);
	
	nw = N / Width;
	nh = N / Height;

	A = (double*)malloc(nw*nh * sizeof(double));
	B = (double*)malloc(nw*nh * sizeof(double));
	C = (double*)malloc(nw*nh * sizeof(double));
	
	work1 = (double*)malloc(nh*K*sizeof(double));
	work2 = (double*)malloc(nw*K*sizeof(double));

	initMat();
	
	int rowid,colid;
	MPI_Comm_Rank(comm_row,&rowid);
	MPI_Comm_Rank(comm_col,&colid);

	MPI_Barrier(MPI_COMM_WORLD);
	double t1 = MPI_Wtime();

	// assume A is transposed
	for(int i = 0; i < Height; i++){
		for(int j = 0; j < Width; j++){
			for(int k = 0; i < K; k++){
				// broadcast a within my row
				broadCast(comm_row,i,rowid,A+k*nh,nh*K,work1);
				// braodcast b within my col
				broadCast(comm_col,j,colid,B+k*nw,nw*K,work2);
				// C(i,j) = C(i,j) + ab
				myDGEMM(nh,nw,K,work1,work2,C);
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double MPI_Wtime();
	if(myid == 0){
		printf("elapsed time = %f sec\n",t2-t1);
	}

	// algo check
	double eps = 1e-12;
	for(int i = 0; i < nw*nh; i++){
		if(fabs(C[i]-N) > eps){
			printf("global MMM wrong\n");
			break;
		}
	}
	
	free(A);
	free(B);
	free(C);
	MPI_Finalize();
	return 0;
}
