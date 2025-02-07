// Perform selection sort in parallel

#include <stdio.h>

__device__ int getGTID() {
    int bnp = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int ntpb = blockDim.x * blockDim.y * blockDim.z;
    int tnb = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int gtid = bnp * ntpb + tnb;
    return gtid;
}

__global__ void selection_sort(int *A, int *B, int n) {
    int gtid = getGTID();

    int pos=0;
    if (gtid < n){
        int data = A[gtid];
        for(int i=0;i<n;i++){
            if (A[i]<A[gtid] || data == A[i] && i<gtid){
                pos++;
            }
        }
        B[pos] = data;
    }
    
}

int main() {
    int *a, *b;
    int n;

    printf("Enter size of array: ");
    scanf("%d", &n);
    a = (int *)malloc(n * sizeof(int));
    b = (int *)malloc(n * sizeof(int));
    printf("Enter array A: ");
    for (int i = 0; i < n; i++)
        scanf("%d", &a[i]);

    int *d_a, *d_b;
    cudaMalloc((void **)&d_a, sizeof(int) * n);
    cudaMalloc((void **)&d_b, sizeof(int) * n);

    cudaMemcpy(d_a, a, sizeof(int) * n, cudaMemcpyHostToDevice);

    selection_sort<<<ceil(n/256.0), 256>>>(d_a, d_b, n);

    cudaMemcpy(b, d_b, sizeof(int) * n, cudaMemcpyDeviceToHost);

    printf("Result B: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", b[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
}
