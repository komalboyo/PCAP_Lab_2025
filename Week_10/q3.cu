
#include <stdio.h>
#include <cuda.h>
          
#define BLOCK_SIZE 4  
 
__global__ void Scan(int *d_in, int *d_out, int *d_blockSums, int n) {
    __shared__ int temp[BLOCK_SIZE];
 
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
   
    temp[tid] = (gid < n) ? d_in[gid] : 0;
    __syncthreads();
 
    
    for (int offset = 1; offset <=tid; offset *= 2) {
        int val = temp[tid - offset] ;
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }
 
    if (tid == BLOCK_SIZE - 1) {
        d_blockSums[blockIdx.x] = temp[tid];
        printf("block %d : %d\n", blockIdx.x, d_blockSums[blockIdx.x]);
    }
    
 
    if (gid < n) d_out[gid] = temp[tid];
}
__global__ void addBlockOffsets(int *d_out, int *d_blockSums, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
     int i;
    if (blockIdx.x > 0 && gid < n) {
      for(i=0;i<blockIdx.x;i++)
        d_out[gid] += d_blockSums[i];
    }
}
 
int main() {
    printf("Enter length of array: ");
    int N;
    scanf("%d",&N);
    //int h_in[N];
    int * h_in = (int *)malloc(sizeof(int)*N);
    printf("Enter array: ");
    for(int i=0; i<N; i++)
        scanf("%d",&h_in[i]);

    //int h_out[N];
    int * h_out = (int *)malloc(sizeof(int)*N);
 
    int *d_in, *d_out, *d_blockSums;
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
 
    cudaMalloc(&d_in, N * sizeof(int));
    int err=cudaMalloc(&d_out, N * sizeof(int));
    if(err!=cudaSuccess)
    printf("error\n");
    cudaMalloc(&d_blockSums, numBlocks * sizeof(int));
 
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
 
    Scan<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, d_blockSums, N);
 
    addBlockOffsets<<<numBlocks, BLOCK_SIZE>>>(d_out, d_blockSums, N);
 
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Prefix Sum: ");
    for (int i = 0; i < N; i++) 
        printf("%d ", h_out[i]);
    printf("\n");
 
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_blockSums);
    return 0;
}
