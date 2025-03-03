#include <stdio.h>
#include <cuda_runtime.h>

#define M 3
#define N 3

__global__ void transformMatrix(int *A, int *B, int m, int n) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    if (row < m && col < n) {
        int val = A[row * n + col];
        int power = row + 1;  // Row index determines power
        int result = 1;
        for (int i = 0; i < power; i++) {
            result *= val;
        }
        B[row * n + col] = result;
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
    int A[M * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9}, B[M * N];

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
