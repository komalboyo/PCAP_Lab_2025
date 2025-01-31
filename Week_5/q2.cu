// Add 2 vectors of length N by keeping the number of threads per block as 256, and vary the number of blocks to handle n elements

#include<stdio.h>
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"

__device__ int getGTID(){
	int blockid = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int threadid = blockid * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadid;
}

__global__ void add(int *a, int *b, int *c, int *n){
	int gtid = getGTID();
	if (gtid < *n){
		c[gtid] = a[gtid]+b[gtid];
	}
}

int main(){
	int *a, *b, *c;
	int n;
	printf("Enter the size of the vectors: ");
	scanf("%d", &n);
	int s = n*sizeof(int);

	a = (int *)malloc(s);
	b = (int *)malloc(s);
	c = (int *)malloc(s);

	printf("Enter values of array A: ");
	for (int i=0;i<n;i++){
		scanf("%d", &a[i]);
	}
	printf("Enter values of array B: ");
	for (int i=0;i<n;i++){
		scanf("%d", &b[i]);
	}

	int *d_a, *d_b, *d_c, *d_n;
	cudaMalloc((void **)&d_a, s);
	cudaMalloc((void **)&d_b, s);
	cudaMalloc((void **)&d_c, s);
	cudaMalloc((void **)&d_n, sizeof(int));  // treat it as a pointer only
	
	cudaMemcpy(d_a, a, s, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, s, cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);

	add<<<ceil(n/256.0), 256>>>(d_a, d_b, d_c, d_n);  // needs to be float for ceil
	cudaMemcpy(c, d_c, s, cudaMemcpyDeviceToHost);

	printf("Result C: ");
	for (int i=0;i<n;i++){
		printf("%d  ", c[i]);
	}
	printf("\n");
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}