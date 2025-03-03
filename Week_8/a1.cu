#include <stdio.h>
#include <cuda_runtime.h>

#define M 2
#define N 3

__global__ void transformMatrix(int *A, int *B, int m, int n) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    if (row < m && col < n) {
        int rowSum = 0, colSum = 0;

        // Compute row sum
        for (int j = 0; j < n; j++) {
            rowSum += A[row * n + j];
        }

        // Compute column sum
        for (int i = 0; i < m; i++) {
            colSum += A[i * n + col];
        }

        // Apply transformation
        int val = A[row * n + col];
        if (val % 2 == 0)
            B[row * n + col] = rowSum;  // Even -> Row Sum
        else
            B[row * n + col] = colSum;  // Odd -> Column Sum
    }
}

void printMatrix(int *M, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", M[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    int size = M * N * sizeof(int);
    int A[M * N] = {1, 2, 3, 4, 5, 6}, B[M * N];

    int *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, M);
    transformMatrix<<<1, threadsPerBlock>>>(d_A, d_B, M, N);

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    printf("Resultant Matrix B:\n");
    printMatrix(B, M, N);

    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
