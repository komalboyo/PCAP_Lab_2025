//5)WAP in MPI wheree even ranked process prints the factorial of the rank and odd ranked process prints the fibonacci number for the ranK.

#include <stdio.h>
#include <mpi.h>

void fact(int rank)
{
	int fact=1;
	for(int i=2;i<=rank;i++)
		fact=fact*i;
	printf("P%d: Factorial is %d\n",rank,fact);
}

void fib(int rank)
{
	int a=0,b=1;
	int fib=rank; //takes care of cases where rank=0 or 1
	for(int i=2;i<=rank;i++) //does not calculate fib for 0 or 1
	{
		fib=a+b;
		a=b;
		b=fib;
	}
	printf("P%d: Fibonacci number is %d\n",rank,fib);
}

int main(int argc, char* argv[])
{
	int rank;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	if(rank%2)
		fib(rank);
	else
		fact(rank);
	MPI_Finalize();
	return 0;
}