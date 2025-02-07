// Program to perform convolution operation on 1D array N of size width using a mask array M of size mask_width to produce result P of size width

#include <stdio.h>

__device__ int getGTID() {
    int bnp = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int ntpb = blockDim.x * blockDim.y * blockDim.z;
    int tnb = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int gtid = bnp * ntpb + tnb;
    return gtid;
}

__global__ void convolution_1d(int *N, int *M, int *P, int width, int mask_width) {
    int gtid = getGTID();

    if (gtid >= width) return;

    int sum = 0;
    int sp = gtid - mask_width / 2;

    for (int i = 0; i < mask_width; i++) {
        if (sp + i >= 0 && sp + i < width) {
            sum += N[sp + i] * M[i];
        }
    }
    P[gtid] = sum;
}

int main() {
    int *n, *p, *m;
    int width, mask_width;

    printf("Enter size of array N: ");
    scanf("%d", &width);

    n = (int *)malloc(width * sizeof(int));
    p = (int *)malloc(width * sizeof(int));

    printf("Enter array N: ");
    for (int i = 0; i < width; i++)
        scanf("%d", &n[i]);

    printf("Enter size of mask M: ");
    scanf("%d", &mask_width);

    m = (int *)malloc(mask_width * sizeof(int));

    printf("Enter mask M: ");
    for (int i = 0; i < mask_width; i++)
        scanf("%d", &m[i]);

    int *d_n, *d_m, *d_p;
    cudaMalloc((void **)&d_n, sizeof(int) * width);
    cudaMalloc((void **)&d_p, sizeof(int) * width);
    cudaMalloc((void **)&d_m, sizeof(int) * mask_width);

    cudaMemcpy(d_n, n, sizeof(int) * width, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, sizeof(int) * mask_width, cudaMemcpyHostToDevice);

    convolution_1d<<<ceil(width/256.0), 256>>>(d_n, d_m, d_p, width, mask_width);

    cudaMemcpy(p, d_p, sizeof(int) * width, cudaMemcpyDeviceToHost);

    printf("Result P: ");
    for (int i = 0; i < width; i++) {
        printf("%d ", p[i]);
    }
    printf("\n");

    cudaFree(d_n);
    cudaFree(d_m);
    cudaFree(d_p);
}
