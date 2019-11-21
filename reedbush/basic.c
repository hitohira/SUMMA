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
// height == Width
int N = (1<<12);
int Height;
int Width;
int K = 4;

#define TIMES 3

int nw,nh;
double *A,*B,*C;
double *work1,*work2;

void initMat(){
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

void broadCast(MPI_Comm comm,int root,int id,double* src,int num,double* dst){
	if(id == root){
		arrayCopy(src,num,dst);
	}
	MPI_Bcast(dst,num,MPI_DOUBLE,root,comm);
}

// A = (n*k)^T, B = k*m, C = n*m
void myDGEMM(int n,int m,int k,double* a,double* b,double* c){
	int p,q,r;
	#pragma omp parallel
	{
		for(p = 0; p < k; p++){
			for(q = 0; q < n; q++){
				#pragma omp for
				for(r = 0; r < m; r++){
					c[q*n+r] += a[p*n+q] * b[p*m+r];
				}
			}
		}
	}
}

int myDGEMMok(){
	double a[16];
	double b[16];
	double c[16],d[16];
	for(int i = 0; i < 16; i++){
		a[i] = 1.0 * rand() / RAND_MAX;
		b[i] = 1.0 * rand() / RAND_MAX;
		c[i] = 0.0;
		d[i] = 0.0;
	}
	myDGEMM(4,4,4,a,b,d);  // assume A is transposed
	for(int i = 0; i < 4; i++){
		for(int j = i+1; j < 4; j++){
			double tmp = a[i*4+j];
			a[i*4+j] = a[j*4+i];
			a[j*4+i] = tmp;
		}
	}
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			for(int k = 0; k < 4; k++){
				c[i*4+j] += a[i*4+k] * b[k*4+j];
			}
		}
	}
	double eps = 1e-12;
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			if(fabs(c[i*4+j]-d[i*4+j]) > eps){
				printf("%f %f %d %d\n",c[i*4+j],d[i*4+j],i,j);
				return 0;
			}
		}
	}
	return 1;
}

int is_pow2(int _n){
	while(_n > 1){
		if(_n % 2 != 0){
			return 0;
		}
		_n /= 2;
	}
	return 1;
}

int main(int argc,char** argv){
	int myid,numproc;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numproc);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);

	int _nnum = (int)sqrt(numproc);
	if(_nnum * _nnum != numproc && is_pow2(_nnum)){
		printf("not sqrt # of proc\n");
		goto fine;
	}
	Width = Height = _nnum;


	// test local MatMatMul
	if(!myDGEMMok() && myid == 0){
		printf("local MMM wrong\n");
		goto fine;
	}
	
	MPI_Comm comm_row,comm_col;

	MPI_Comm_split(MPI_COMM_WORLD,myid % Width, myid / Width,&comm_row);
	MPI_Comm_split(MPI_COMM_WORLD,myid / Width, myid % Width,&comm_col);
	
	nw = N / Width;
	nh = N / Height;

	A = (double*)malloc(nw*nh * sizeof(double));
	B = (double*)malloc(nw*nh * sizeof(double));
	C = (double*)malloc(nw*nh * sizeof(double));
	
	work1 = (double*)malloc(nh*nw/K*sizeof(double));
	work2 = (double*)malloc(nw*nh/K*sizeof(double));

	
	int rowid,colid;
	MPI_Comm_rank(comm_row,&rowid);
	MPI_Comm_rank(comm_col,&colid);

	double t_sum = 0.0;
	for(int tm = 0; tm <= TIMES; tm++){
		initMat();

		MPI_Barrier(MPI_COMM_WORLD);
		double t1 = MPI_Wtime();


		// assume A is transposed
		for(int i = 0; i < Height; i++){
			for(int k = 0; k < K; k++){
				// broadcast a within my row
				broadCast(comm_row,i,rowid,A+k*nh*(nw/K),nh*(nw/K),work1);
				// braodcast b within my col
				broadCast(comm_col,i,colid,B+k*nw*(nh/K),nw*(nh/K),work2);
				// C(i,j) = C(i,j) + ab
				myDGEMM(nh,nw,nw/K,work1,work2,C);
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
		double t2 = MPI_Wtime();
		
		if(tm != 0){
			t_sum += t2 - t1;
		}
	}
	if(myid == 0){
		printf("elapsed time = %f sec\n",t_sum/TIMES);
	}

	// algo check
	double eps = 1e-12;
	for(int i = 0; i < nw*nh; i++){
		if(fabs(C[i]-N) > eps && myid == 0){
			printf("%f %d\n",C[i],N);
			printf("global MMM wrong\n");
			break;
		}
	}
	free(work1);
	free(work2);
	free(A);
	free(B);
	free(C);
fine:
	MPI_Finalize();
	return 0;
}
