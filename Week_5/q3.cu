// Process a 1d array containing angles in radians to generate sine of angles in output.

#include<stdio.h>

__global__ void sine_angles(double *a, double *b, int *n){
	int gtid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gtid < *n){
		b[gtid] = sin(a[gtid]);
	}
}

int main(){
	double *a, *b;
	int n;
	printf("Enter the size of the array: ");
	scanf("%d", &n);
	int s = n*sizeof(double);

	a = (double *)malloc(s);
	b = (double *)malloc(s);

	printf("Enter angles in rad: ");
	for (int i=0;i<n;i++){
		scanf("%lf", &a[i]);
	}

	double *d_a, *d_b;
	int *d_n;
	cudaMalloc((void **)&d_a, s);
	cudaMalloc((void **)&d_b, s);
	cudaMalloc((void **)&d_n, sizeof(int));  // treat it as a pointer only
	
	cudaMemcpy(d_a, a, s, cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);

	sine_angles<<<ceil(n/256.0), 256>>>(d_a, d_b, d_n);  // needs to be float for ceil
	cudaMemcpy(b, d_b, s, cudaMemcpyDeviceToHost);

	printf("Result : \n");
	for (int i=0;i<n;i++){
		printf("sine(%lf) -> %lf  \n", a[i], b[i]);
	}
	printf("\n");
	cudaFree(d_a);
	cudaFree(d_b);
}