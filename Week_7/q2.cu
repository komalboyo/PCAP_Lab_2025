// Write a cuda program that reads a string S and produces the 
// string RS as follows:
// Each work item copies required number of characters from S to RS

#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>

__global__ void generate_rs(char *s, char *rs, int len_s) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len_s) return;

    // Calculate start position for each segment
    int start_pos = 0;
    for (int i = 0; i < idx; i++) {
        start_pos += (len_s - i);
    }

    // Copy substring into correct position
    for (int j = 0; j < (len_s - idx); j++) {
        rs[start_pos + j] = s[j];
    }
}

int main() {
    char h_s[] = "pcap";
    int len_s = strlen(h_s);

    // Compute size of RS string
    int len_rs = (len_s * (len_s + 1)) / 2;
    char h_rs[len_rs + 1];  // Extra space for null terminator
    h_rs[len_rs] = '\0';   

     char *d_s, *d_rs;
    cudaMalloc((void**)&d_s, len_s * sizeof(char));
    cudaMalloc((void**)&d_rs, len_rs * sizeof(char));

     cudaMemcpy(d_s, h_s, len_s * sizeof(char), cudaMemcpyHostToDevice);

     int threads_per_block = 256;
    int blocks = (len_s + threads_per_block - 1) / threads_per_block;
    generate_rs<<<blocks, threads_per_block>>>(d_s, d_rs, len_s);

     cudaMemcpy(h_rs, d_rs, len_rs * sizeof(char), cudaMemcpyDeviceToHost);

     printf("S  : %s\n", h_s);
    printf("RS : %s\n", h_rs);

     cudaFree(d_s);
    cudaFree(d_rs);

    return 0;
}
