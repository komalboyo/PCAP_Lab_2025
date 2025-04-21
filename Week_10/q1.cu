// Matrix multiplication of 4x4 matrix using Tiling and Shared Memory
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_WIDTH 2
#define TILE_WIDTH 2
#define WIDTH 4

__global__ void MatMulElementThreadShared(int *a, int *b, int *c) {
    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];  // for matA
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];  // for matB

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    int Pvalue = 0;

    for (int m = 0; m < WIDTH / TILE_WIDTH; ++m) {
        Mds[ty][tx] = a[Row * WIDTH + m * TILE_WIDTH + tx];
        Nds[ty][tx] = b[(m * TILE_WIDTH + ty) * WIDTH + Col];

        __syncthreads();  // threads have finished loading their data into shared memory

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();  // current tile completed
    }

    c[Row * WIDTH + Col] = Pvalue;
}

int main() {
    int *matA, *matB, *matProd;
    int *da, *db, *dc;

    printf("\nEnter elements of Matrix A (4x4):\n");
    matA = (int *)malloc(sizeof(int) * WIDTH * WIDTH);
    for (int i = 0; i < WIDTH * WIDTH; ++i)
        scanf("%d", &matA[i]);

    printf("\nEnter elements of Matrix B (4x4):\n");
    matB = (int *)malloc(sizeof(int) * WIDTH * WIDTH);
    for (int i = 0; i < WIDTH * WIDTH; ++i)
        scanf("%d", &matB[i]);

    matProd = (int *)malloc(sizeof(int) * WIDTH * WIDTH);

    cudaMalloc((void **)&da, sizeof(int) * WIDTH * WIDTH);
    cudaMalloc((void **)&db, sizeof(int) * WIDTH * WIDTH);
    cudaMalloc((void **)&dc, sizeof(int) * WIDTH * WIDTH);

    cudaMemcpy(da, matA, sizeof(int) * WIDTH * WIDTH, cudaMemcpyHostToDevice);
    cudaMemcpy(db, matB, sizeof(int) * WIDTH * WIDTH, cudaMemcpyHostToDevice);

    int NumBlocks = WIDTH / BLOCK_WIDTH;
    dim3 grid_conf(NumBlocks, NumBlocks);
    dim3 block_conf(BLOCK_WIDTH, BLOCK_WIDTH);

    MatMulElementThreadShared<<<grid_conf, block_conf>>>(da, db, dc);

    cudaMemcpy(matProd, dc, sizeof(int) * WIDTH * WIDTH, cudaMemcpyDeviceToHost);

    printf("\nResult of Matrix Multiplication:\n");
    for (int i = 0; i < WIDTH; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            printf("%6d ", matProd[i * WIDTH + j]);
        }
        printf("\n");
    }
}
