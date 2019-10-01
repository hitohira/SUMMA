#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void mul_p(int n,double* a,double*b,double* c){
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			c[i*n+j] = 0.0;
			for(int k = 0; k < n; k++){
				c[i*n+j] += a[i*n+k] * b[k*n+j];
			}
		}
	}
}

void mul(int n,double* a,double*b,double* c){
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			c[i*n+j] = 0.0;
			#pragma omp parallel for
			for(int k = 0; k < n; k++){
				c[i*n+j] += a[i*n+k] * b[k*n+j];
			}
		}
	}
}

double getT(struct timeval t1,struct timeval t2){
	return (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) * 0.000001;
}

int main(){
	double *A,*B,*C;
	int n = 1<<9;
	A = (double*)malloc(n*n*sizeof(double));
	B = (double*)malloc(n*n*sizeof(double));
	C = (double*)malloc(n*n*sizeof(double));

	for(int i = 0; i < n*n; i++){
		A[i] = 1.0;
		B[i] = 1.0;
	}
	mul(n,A,B,C);
	printf("%f\n",C[rand()%(n*n)]);

	struct timeval t1,t2;
	gettimeofday(&t1,NULL);

	mul(n,A,B,C);

	gettimeofday(&t2,NULL);
	printf("t=%f\n",getT(t1,t2));
	printf("%f\n",C[rand()%(n*n)]);


	gettimeofday(&t1,NULL);

	mul_p(n,A,B,C);

	gettimeofday(&t2,NULL);
	printf("t=%f\n",getT(t1,t2));
	printf("%f\n",C[rand()%(n*n)]);

	return 0;
}
