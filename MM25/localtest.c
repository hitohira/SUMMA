#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "MM25.h"

double A[512*512],B[512*512],C[512*512];

int main(){
	for(int i = 0; i < 512; i++){
		A[i] = 1;
		B[i] = 1;
		C[i] = 0;
	}
	localMul(512,A,B,C);

	for(int i = 0; i < 512*512;i++){
		if(fabs(C[i]-512) > 1e4){
			printf("bad %f\n",C[i]);
			return 1;
		}
	}
	printf("OK\n");
	return 0;
}
