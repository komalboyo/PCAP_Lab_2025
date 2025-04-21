#include <stdio.h>
#define MAX_MASK_WIDTH 10
__constant__ int gM[MAX_MASK_WIDTH];

__device__ int get_gtid()
{
    int bng = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int ntpb = blockDim.x * blockDim.y * blockDim.z;
    int tnb = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int gtid = bng * ntpb + tnb;
    return gtid;
}

__global__ void conv1d(int * N,int * P,int width,int mask_width)
{
    int gtid = get_gtid();
    if(gtid < width)
    {
        int sum=0;
        int sp = gtid - mask_width/2; //starting point to align mask and arr
        for(int i=0; i < mask_width; i++)
        {
            if(sp+i >= 0 && sp+i < width) // to avoid avoid access index < 0 and index > width - 1
                sum+= N[sp + i] * gM[i];
        }
        P[gtid] = sum;
    }
}

int main()
{
    int width;
    int mask_width;
    
    printf("Enter width: ");
    scanf("%d",&width);
    printf("Enter mask_width: ");
    scanf("%d",&mask_width);

    int nsize =  width * sizeof(int);
    int msize = mask_width * sizeof(int);
    int *N = (int *)malloc(sizeof(int)*width);
    int *M = (int *)malloc(sizeof(int)*mask_width);
    int *P = (int *)malloc(sizeof(int)*width);
    // int N[width], M[mask_width], P[width];
    int *d_N, *d_P;

    printf("Enter elements in N: ");
    for(int i=0; i< width; i++)
        scanf("%d",&N[i]);

    printf("Enter elements in M: ");
    for(int i=0; i< mask_width; i++)
        scanf("%d",&M[i]);

    cudaMalloc((void**)&d_N , nsize);
    cudaMalloc((void**)&d_P , nsize);

    cudaMemcpy(d_N,N,nsize,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gM,M,msize);

    dim3 blk(2,2,1); //i.e 4 threads per block
    conv1d<<<ceil(width/4.0),blk>>>(d_N,d_P,width,mask_width);

    cudaMemcpy(P,d_P,nsize,cudaMemcpyDeviceToHost);

    printf("Result: ");
	for(int i=0; i<width; i++)
	{
		printf("%d ",P[i]);
	}
	printf("\n");
}
