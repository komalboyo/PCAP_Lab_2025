/*
Perform parallel sparse matrix multiplication using CSR storage
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void parallelspv(int*data,int*col_index,int*row_ptr,int*x,int*y,int num_rows)
{
    int row = threadIdx.x + blockDim.x*blockIdx.x;

    if(row < num_rows)
    {
        int dot = 0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row+1];

        for(int i= row_start; i<row_end; i++)
            dot += data[i]*x[col_index[i]];
        
        y[row]=dot;
    }
}

int main()
{
    printf("Enter dimensions of input 2D Vector: ");
    int m,n;
    scanf("%d %d",&m,&n);
    int input[m][n];
    printf("Enter the 2D Vector: \n");
    int k=0;
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            scanf("%d",&input[i][j]);
            if(input[i][j]!=0)
                k++;
        }
    }
    int data[k];
    int row_ptr[m+1];
    int col_index[k];
    int l=0;

    for(int i=0;i<m;i++) //converting to csr format
    {
        row_ptr[i]=l;
        for(int j=0;j<n;j++)
        {
            if(input[i][j]!=0)
            {
                data[l] = input[i][j];
                col_index[l]= j;
                l++;
            }
        }
    }
    row_ptr[m]= k;

    printf("CSR: ");
    for(int i=0;i<k;i++)
        printf("%d ",data[i]);
    
    printf("\nrow_ptr: ");
    for(int i=0;i<=m;i++)
        printf("%d ",row_ptr[i]);
    
    printf("\ncol_index: ");
    for(int i=0;i<k;i++)
        printf("%d ",col_index[i]);

    printf("\nEnter Vector x: \n");
    int x[m];
    for(int i=0;i<m;i++)
        scanf("%d",&x[i]);

    int y[m];
    int *d_data, *d_row_ptr, *d_col_index, *d_x, *d_y;

    cudaMalloc((void**)&d_data,k*sizeof(int));
    cudaMalloc((void**)&d_row_ptr,(m+1)*sizeof(int));
    cudaMalloc((void**)&d_col_index,k*sizeof(int));
    cudaMalloc((void**)&d_x,m*sizeof(int));
    cudaMalloc((void**)&d_y,m*sizeof(int));

    cudaMemcpy(d_data,data,k*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr,row_ptr,(m+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_index,col_index,k*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,x,m*sizeof(int),cudaMemcpyHostToDevice);

    //let's do 2 threads per block
    parallelspv<<<(m/2),2>>>(d_data,d_col_index,d_row_ptr,d_x,d_y,m);

    cudaMemcpy(y,d_y,m*sizeof(int),cudaMemcpyDeviceToHost);

    printf("Result: \n");
    for(int i=0;i<m;i++)
    {
        printf("%d\n",y[i]);
    }

    cudaFree(d_data);
    cudaFree(d_row_ptr);
    cudaFree(d_col_index);
    cudaFree(d_x);
    cudaFree(d_y);

}
