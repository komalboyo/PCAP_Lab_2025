// Write a program to multiply 2 matrices
// a) each row of result is computed by 1 thread
// b) each column of result is computed by 1 thread
// c) each element of result is computed by 1 thread

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void multiplyRows(int *A, int *B, int *C, int wa, int wb) {
    int r = threadIdx.x;
    int sum=0;
    for (int c = 0; c < wb; c++) {
        sum=0;
        for (int k = 0; k < wa; k++) {
            sum += A[r * wa + k] * B[k * wb + c];
        }
        C[r*wb + c] = sum;
    }
}

__global__ void multiplyCols(int *A, int *B, int *C, int ha, int wa) {
    int c = threadIdx.x;
    int sum=0;
    int wb = blockDim.x;
    for (int r = 0; r < ha; r++) {
        sum = 0;
        for (int k = 0; k < wa; k++) {
            sum += A[r * wa + k] * B[k * wb + c];
        }
        C[r*wb+c]=sum;
    }
}

__global__ void multiplyElements(int *A, int *B, int *C, int wa) {
    int r = threadIdx.y;
    int c = threadIdx.x;
    int wb = blockDim.x;
    int sum=0;
    for (int k=0;k<wa;k++){
        sum+=A[r*wa+k]*B[k*wb+c];
    }
    C[r*wb+c] = sum;
}

void printMatrix(int *M, int ha, int wb) {
    for (int i = 0; i < ha; i++) {
        for (int j = 0; j < wb; j++) {
            printf("%d ", M[i * wb + j]);
        }
        printf("\n");
    }
}

int main() {
    int ha = 3, wa = 2;  // Matrix A: 3x2
    int hb = 2, wb = 4;  // Matrix B: 2x4

    int size_A = ha * wa * sizeof(int);
    int size_B = hb * wb * sizeof(int);
    int size_C = ha * wb * sizeof(int);

    int *A = (int *)malloc(size_A);
    int *B = (int *)malloc(size_B);
    int *C = (int *)malloc(size_C);

    // Initialize matrices A and B
    for (int i = 0; i < ha * wa; i++) {
        A[i] = i + 1;
    }
    for (int i = 0; i < hb * wb; i++) {
        B[i] = (i + 1) * 2;
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    printf("Matrix A:\n");
    printMatrix(A, ha, wa);
    printf("\nMatrix B:\n");
    printMatrix(B, hb, wb);
    printf("\nMatrix C (Result):\n");

    // Uncomment the kernel you want to use

    // Case (a): Each row computed by one thread
    // multiplyRows<<<1, ha>>>(d_A, d_B, d_C, wa, wb);

    // Case (b): Each column computed by one thread
    multiplyCols<<<1, wb>>>(d_A, d_B, d_C, ha, wa);

    // Case (c): Each element computed by one thread
    // dim3 threadsPerBlock(wb, ha);
    // multiplyElements<<<1, threadsPerBlock>>>(d_A, d_B, d_C, wa);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    printMatrix(C, ha, wb);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
