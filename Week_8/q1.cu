#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addRows(int *A, int *B, int *C, int n, int m) {
    int row = threadIdx.x;
    if (row < n) {
        for (int j = 0; j < m; j++) {
            C[row * m + j] = A[row * m + j] + B[row * m + j];
        }
    }
}

__global__ void addCols(int *A, int *B, int *C, int n, int m) {
    int col = threadIdx.x;
    if (col < m) {
        for (int i = 0; i < n; i++) {
            C[i * m + col] = A[i * m + col] + B[i * m + col];
        }
    }
}

__global__ void addElements(int *A, int *B, int *C, int n, int m) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < n && col < m) {
        int idx = row * m + col;
        C[idx] = A[idx] + B[idx];
    }
}

void printMatrix(int *M, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%d ", M[i * m + j]);
        }
        printf("\n");
    }
}

int main() {
    int n = 3;  // Number of rows
    int m = 4;  // Number of columns

    int size = n * m * sizeof(int);
    int *A = (int *)malloc(size);
    int *B = (int *)malloc(size);
    int *C = (int *)malloc(size);

    // Initialize matrices
    for (int i = 0; i < n * m; i++) {
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
    printMatrix(A, n, m);
    printf("\n+\n");
    printMatrix(B, n, m);
    printf("\n=\n");

    // Uncomment the kernel you want to use

    // Case (a): Each row computed by one thread
    // addRows<<<1, n>>>(d_A, d_B, d_C, n, m);

    // Case (b): Each column computed by one thread
    // addCols<<<1, m>>>(d_A, d_B, d_C, n, m);

    // Case (c): Each element computed by one thread
    dim3 threadsPerBlock(m, n);
    addElements<<<1, threadsPerBlock>>>(d_A, d_B, d_C, n, m);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printMatrix(C, n, m);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
