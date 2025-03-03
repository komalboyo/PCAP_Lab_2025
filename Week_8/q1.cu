#include <stdio.h>
#include <cuda_runtime.h>

#define N 3 // Define matrix size (N x N for simplicity)

__global__ void addRows(int *A, int *B, int *C, int n) {
    int row = threadIdx.x;
    if (row < n) {
        for (int j = 0; j < n; j++) {
            C[row * n + j] = A[row * n + j] + B[row * n + j];
        }
    }
}

__global__ void addCols(int *A, int *B, int *C, int n) {
    int col = threadIdx.x;
    if (col < n) {
        for (int i = 0; i < n; i++) {
            C[i * n + col] = A[i * n + col] + B[i * n + col];
        }
    }
}

__global__ void addElements(int *A, int *B, int *C, int n) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < n && col < n) {
        int idx = row * n + col;
        C[idx] = A[idx] + B[idx];
    }
}

void printMatrix(int *M, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", M[i * n + j]);
        }
        printf("\n");
    }
}

int main() {
    int size = N * N * sizeof(int);
    int A[N * N], B[N * N], C[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    printf("Original Matrices:\n");
    printMatrix(A, N);
    printf("\n+\n");
    printMatrix(B, N);
    printf("\n=\n");

    // Uncomment the kernel you want to use

    // Case (a): Each row computed by one thread
    // addRows<<<1, N>>>(d_A, d_B, d_C, N);

    // Case (b): Each column computed by one thread
    // addCols<<<1, N>>>(d_A, d_B, d_C, N);

    // Case (c): Each element computed by one thread
    dim3 threadsPerBlock(N, N);
    addElements<<<1, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printMatrix(C, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
