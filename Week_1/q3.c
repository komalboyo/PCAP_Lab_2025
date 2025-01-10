//3) WAP in MPI to simulate simple calculator. Perform each operation using diff operations in parallel

#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
	int const a=23,b=5;
	int c,rank;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	switch(rank)
	{
		case 0: c=a+b;
			printf("a+b=%d\n",c);
			break;
		case 1: c=a-b;
			printf("a-b=%d\n",c);
			break;
		case 2: c=a*b;
			printf("a*b=%d\n",c);
			break;
		case 3: c=a/b;
			printf("a/b=%d\n",c);
			break;
		case 4: c=a%b;
			printf("a%%b=%d\n",c); //%% allows us to print %
			break;	
		default: printf("No more operations left\n");		
	}
	return 0;
}