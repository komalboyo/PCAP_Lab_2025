// Week 2 Q3

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char*argv[])
{
	int rank,size;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Status status;
	if(rank==0){

		int*buf;
		int bsize;
		bsize = MPI_BSEND_OVERHEAD+(size-1)*sizeof(int);
		buf=(int*)malloc(bsize);
		printf("Enter %d numbers: ",size-1);
		int data[size-1];
		for(int i=0;i<size-1;i++)
			scanf("%d",&data[i]);
		
		MPI_Buffer_attach(buf,bsize);
		for(int i=0;i<size-1;i++)
			MPI_Bsend(data+i,1,MPI_INT,i+1,i+1,MPI_COMM_WORLD);
		MPI_Buffer_detach(&buf,&bsize);
	}

	else{
		int num;
		MPI_Recv(&num,1,MPI_INT,0,rank,MPI_COMM_WORLD,&status);
		if(rank%2)
		{
			int cubed;
			cubed=num*num*num;
			printf("P%d: Cube of %d = %d\n",rank,num,cubed);
		}

		else{
			int sq;
			sq=num*num;
			printf("P%d: Square of %d = %d\n",rank,num,sq);
		}
	}
	MPI_Finalize();
	return 0;
}