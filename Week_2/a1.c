// Week 2 additional 1

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc,char* argv[])
{
    int rank,size;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;

    if(rank==0)
    {
        int data[100];
        int x, dest=0;
        printf("Enter %d elements: ", size);
        for (int i=0;i<size;i++){
            scanf("%d", &data[i]);
            MPI_Send(&data[i],1,MPI_INT,dest++,1,MPI_COMM_WORLD);
            printf("Sent %d from P%d to P%d\n",data[i], rank, dest-1);
        }

        MPI_Recv(&x,1,MPI_INT,0,1,MPI_COMM_WORLD,&status);
        printf("Received %d in Process %d\n",x,rank);
    }
    else
    {
        int x;
        MPI_Recv(&x,1,MPI_INT,0,1,MPI_COMM_WORLD,&status);
        printf("Received %d in process %d\n",x,rank);
        //x++;
        //MPI_Ssend(&x,1,MPI_INT,t,1,MPI_COMM_WORLD);
        //printf("Sent %d to Process %d\n",x,t);
    }
    MPI_Finalize();
    return 0;
}
