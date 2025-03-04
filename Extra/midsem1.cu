/* Given an NXN matrix mat, where N is an even number and N >= 4, write a CUDA program to
 find major diagonal sums of partial matrices in the four different quadrants (vertical and horizontal lines are
drawn to partition the matrix into 4 quadrants) of mat in parallel as shown below in sample Input/Output.
 Use 2X2 grid with 1 thread per block. The first parameter to the kernel is the matrix and the second is an 1D
 array of size 4 to store 4 different diagonal sums computed by 4 different threads of 4 different blocks. Pass
 any other parameters if necessary. Read N and mat contents in the host code.  Display the matrix mat and the
 diagonal sums in the host code. The code should work for any value of N satisfying the condition mentioned
 above.  Use dynamic allocation for all the host arrays.*/

#include <stdio.h>
#include <stdlib.h>

#define BLOCKS_PER_DIM 2 // 2x2 grid

__global__ void compute_diagonal_sums(int *mat, int *sums, int N) {
    int bx = blockIdx.x; // Block index (0 or 1 for each dimension)
    int by = blockIdx.y;
    
    int quadrant_id = by * 2 + bx; // Assign quadrant ID (0 to 3)
    int start_row = (by == 0) ? 0 : N / 2;
    int start_col = (bx == 0) ? 0 : N / 2;
    
    int diag_sum = 0;
    for (int i = 0; i < N / 2; i++) {
        diag_sum += mat[(start_row + i) * N + (start_col + i)];
    }

    sums[quadrant_id] = diag_sum;
}

int main() {
    int N;
    printf("Enter an even N (>=4): ");
    scanf("%d", &N);
    
    if (N < 4 || N % 2 != 0) {
        printf("N must be an even number >= 4.\n");
        return 1;
    }

    int *h_mat = (int*)malloc(N * N * sizeof(int));
    int *h_sums = (int*)malloc(4 * sizeof(int));
    
    // Fill the matrix with sample values
    printf("Enter %d x %d matrix elements:\n", N, N);
    for (int i = 0; i < N * N; i++) {
        scanf("%d", &h_mat[i]);
    }

    // Print matrix
    printf("Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_mat[i * N + j]);
        }
        printf("\n");
    }

    // Allocate memory on GPU
    int *d_mat, *d_sums;
    cudaMalloc((void**)&d_mat, N * N * sizeof(int));
    cudaMalloc((void**)&d_sums, 4 * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_mat, h_mat, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 grid(BLOCKS_PER_DIM, BLOCKS_PER_DIM); // 2x2 grid
    compute_diagonal_sums<<<grid, 1>>>(d_mat, d_sums, N);

    // Copy results back
    cudaMemcpy(h_sums, d_sums, 4 * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Diagonal sums of quadrants:\n");
    for (int i = 0; i < 4; i++) {
        printf("Quadrant %d: %d\n", i, h_sums[i]);
    }

    // Free memory
    free(h_mat);
    free(h_sums);
    cudaFree(d_mat);
    cudaFree(d_sums);

    return 0;
}
