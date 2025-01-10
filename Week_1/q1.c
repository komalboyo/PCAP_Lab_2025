//1) Write a simple MPI program to find out pow(x,rank) for all the [rpcesses where x is the integer constant and rank is the rank of the process

#include <stdio.h>
#include <mpi.h>

void power(int const x, int rank)
{
	int r=1;
	for(int i=1;i<=rank;i++)
	 r=r*x;
	printf("P%d: power of (%d,%d) is %d\n",rank,x,rank,r);
}

int main(int argc, char* argv[])
{
	int const x = 2;
	int rank;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	power(x,rank);
	MPI_Finalize();
	return 0;
}

//WHOLE CODE RUNS FOR ALL THE PROCESSES NOT JUST FROM INIT ONWARDS