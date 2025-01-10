//2) WAP in MPI where even ranked process prints "Hello" & odd raned process prints "World"

#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
	int rank;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	if(rank%2)
		printf("P%d: World\n",rank);
	else
		printf("P%d: Hello\n",rank);
	MPI_Finalize();
	return 0;
}