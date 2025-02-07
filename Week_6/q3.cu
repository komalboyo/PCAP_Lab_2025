//  Program to perform even odd transposiiton sort in parallel

#include <stdio.h>

__device__ int getGTID() {
    int bnp = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int ntpb = blockDim.x * blockDim.y * blockDim.z;
    int tnb = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int gtid = bnp * ntpb + tnb;
    return gtid;
}

__global__ void even_kernel(int *A, int n) {
    int gtid = getGTID();
    if (gtid < n && gtid%2 == 0){
        if (gtid+1 <= n-1 && A[gtid]> A[gtid+1]){
            int temp = A[gtid];
            A[gtid] = A[gtid+1];
            A[gtid+1] = temp;
        }
    }
}

__global__ void odd_kernel(int *A, int n) {
    int gtid = getGTID();
    if (gtid < n && gtid%2 == 1){
        if (gtid+1 <= n-1 && A[gtid]> A[gtid+1]){
            int temp = A[gtid];
            A[gtid] = A[gtid+1];
            A[gtid+1] = temp;
        }
    }
}

int main() {
    int *a;
    int n;
    printf("Enter size of array: ");
    scanf("%d", &n);
    a = (int *)malloc(n * sizeof(int));
    printf("Enter array A: ");
    for (int i = 0; i < n; i++)
        scanf("%d", &a[i]);

    int *d_a;
    cudaMalloc((void **)&d_a, sizeof(int) * n);

    cudaMemcpy(d_a, a, sizeof(int) * n, cudaMemcpyHostToDevice);
    for (int i=0;i<n/2;i++){
        odd_kernel<<<1, n>>>(d_a, n);
        even_kernel<<<1, n>>>(d_a, n);
    }

    cudaMemcpy(a, d_a, sizeof(int) * n, cudaMemcpyDeviceToHost);

    printf("Result : ");
    for (int i = 0; i < n; i++) { 
        printf("%d ", a[i]);
    }
    printf("\n");

    cudaFree(d_a);
}
