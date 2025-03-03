#include <stdio.h>
#include <cuda_runtime.h>

#define N 3

__device__ int factorial(int num) {
    int fact = 1;
    for (int i = 1; i <= num; i++)
        fact *= i;
    return fact;
}

__device__ int sumOfDigits(int num) {
    int sum = 0;
    while (num > 0) {
        sum += num % 10;
        num /= 10;
    }
    return sum;
}

__global__ void transformMatrix(int *A, int *B, int n) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    if (row < n && col < n) {
        if (row == col) {
            B[row * n + col] = 0; // Principal diagonal -> 0
        } else if (row < col) {
            B[row * n + col] = factorial(A[row * n + col]); // Above diagonal -> Factorial
        } else {
            B[row * n + col] = sumOfDigits(A[row * n + col]); // Below diagonal -> Sum of digits
        }
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
    int A[N * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9}, B[N * N];

    int *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, N);
    transformMatrix<<<1, threadsPerBlock>>>(d_A, d_B, N);

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    printf("Transformed Matrix B:\n");
    printMatrix(B, N);

    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
