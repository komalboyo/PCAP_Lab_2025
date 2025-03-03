#include <stdio.h>
#include <cuda_runtime.h>

#define M 4
#define N 4

__global__ void replaceInnerWithOnesComplement(int *A, int *B, int m, int n) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    if (row < m && col < n) {
        if (row == 0 || row == m - 1 || col == 0 || col == n - 1)
            B[row * n + col] = A[row * n + col];  // Keep border elements same
        else
            B[row * n + col] = ~A[row * n + col]; // 1's complement for inner elements
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
    int A[M * N] = {1, 2, 3, 4, 6, 5, 8, 3, 2, 4, 10, 1, 9, 1, 2, 5}, B[M * N];

    int *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, M);
    replaceInnerWithOnesComplement<<<1, threadsPerBlock>>>(d_A, d_B, M, N);

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    printf("Resultant Matrix B:\n");
    printMatrix(B, M, N);

    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
