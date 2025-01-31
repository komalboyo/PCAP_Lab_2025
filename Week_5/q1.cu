// Writ a program in CUDA to add 2 vectors of length N using 
// (a) block number as N
// (b) 1 block N threads


#include<stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ int getGTID(){
	int blockid = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int threadid = blockid * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadid;
}

__global__ void add_a(int *a, int *b, int *c){
	int i = blockIdx.x;
	c[i] = a[i]+b[i];
}

__global__ void add_b(int *a, int *b, int *c){
	int i = threadIdx.x;
	c[i] = a[i]+b[i];
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

	int *d_a, *d_b, *d_c;
	cudaMalloc((void **)&d_a, s);
	cudaMalloc((void **)&d_b, s);
	cudaMalloc((void **)&d_c, s);

	cudaMemcpy(d_a, a, s, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, s, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, s, cudaMemcpyHostToDevice);

	add_b<<<1, n>>>(d_a, d_b, d_c);
	cudaMemcpy(c, d_c, s, cudaMemcpyDeviceToHost);

	printf("Result C by method b: ");
	for (int i=0;i<n;i++){
		printf("%d  ", c[i]);
	}
	cudaFree(d_c);

	add_a<<<n, 1>>>(d_a, d_b, d_c);
	cudaMemcpy(c, d_c, s, cudaMemcpyDeviceToHost);

	printf("Result C by method a: ");
	for (int i=0;i<n;i++){
		printf("%d  ", c[i]);
	}
	printf("\n");
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}