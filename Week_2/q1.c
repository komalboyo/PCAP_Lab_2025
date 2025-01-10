// Week 2 Q1

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char*argv[])
{
    int rank,size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    MPI_Status status;
    if(rank==0)
    {
        char data[10];
        printf("Enter word: ");
        scanf("%[^\n]c",data);
        char new[strlen(data)+1];
        MPI_Ssend(data,strlen(data)+1,MPI_CHAR,1,1,MPI_COMM_WORLD);
        MPI_Recv(new,strlen(data)+1,MPI_CHAR,1,1,MPI_COMM_WORLD,&status);   //size is +1 since you have to send \0 also
        sleep(2);
        printf("P0 says: P1 sent %s\n",new);
    }

    else if(rank==1){
        char data[10];
        MPI_Recv(data,10,MPI_CHAR,0,1,MPI_COMM_WORLD,&status);
        printf("P1 says: P0 sent %s\n",data);
        int len=strlen(data);
        for(int i=0;i<len;i++)
        {
            if(data[i]>=65 && data[i]<=90)
                data[i]= data[i]+ 32;

            else if(data[i]>=97 && data[i]<=122)
                data[i]= data[i]- 32;
        }
        MPI_Ssend(data,strlen(data)+1,MPI_CHAR,0,1,MPI_COMM_WORLD);

    }
    MPI_Finalize();
    return 0;
}