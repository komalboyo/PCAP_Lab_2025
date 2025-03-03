#include <stdio.h>
#include <cuda_runtime.h>

#define M 2
#define N 4

__global__ void generateString(char *A, int *B, char *output, int *outIndex, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        char ch = A[row * n + col];
        int repeat = B[row * n + col];
        int startIdx = atomicAdd(outIndex, repeat);

        for (int i = 0; i < repeat; i++) {
            output[startIdx + i] = ch;
        }
    }
}

int main() {
    char A[M * N] = {'p', 'C', 'a', 'P', 'e', 'X', 'a', 'M'};
    int B[M * N] = {1, 2, 4, 3, 2, 4, 3, 2};

    int totalSize = 0;
    for (int i = 0; i < M * N; i++) {
        totalSize += B[i];
    }

    char *d_A, *d_output;
    int *d_B, *d_outIndex;
    int outIndex = 0;

    cudaMalloc(&d_A, M * N * sizeof(char));
    cudaMalloc(&d_B, M * N * sizeof(int));
    cudaMalloc(&d_output, totalSize * sizeof(char));
    cudaMalloc(&d_outIndex, sizeof(int));

    cudaMemcpy(d_A, A, M * N * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outIndex, &outIndex, sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, M);
    generateString<<<1, threadsPerBlock>>>(d_A, d_B, d_output, d_outIndex, M, N);

    char *output = (char *)malloc(totalSize * sizeof(char));
    cudaMemcpy(output, d_output, totalSize * sizeof(char), cudaMemcpyDeviceToHost);

    printf("Output String STR: ");
    for (int i = 0; i < totalSize; i++) {
        printf("%c", output[i]);
    }
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_output);
    cudaFree(d_outIndex);
    free(output);

    return 0;
}
