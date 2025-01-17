// Read n values in root process, root sends one value to each process, each process finds facorial of number 
// and returns to root process, root gathers factorial and finds sum of it

#include<stdio.h>
#include<mpi.h>
#include<unistd.h>

int fact(int n)
{
    if(n==1||n==0)
        return 1;
    return n*fact(n-1);
}

int main(int argc, char *argv[]){
	int rank, size, N, c, sum=0;
	int A[100], B[100];
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;

	if(rank==0){
        N = size;
        printf("Enter %d Values : \n",N);
        for(int i=0;i<N;i++)
            scanf("%d",&A[i]);
    }

    MPI_Scatter(A,1,MPI_INT,&c,1,MPI_INT,0,MPI_COMM_WORLD);
    printf("Received %d in %d\n",c,rank);
    c = fact(c);

    MPI_Gather(&c,1,MPI_INT,B,1,MPI_INT,0,MPI_COMM_WORLD);

    if(rank==0){
    	sleep(1);	// cuz printing takes time
        printf("Gathered results : \n");
        for(int i=0;i<N;i++){
            printf("%d \n",B[i]);
            sum += B[i];
        }
        printf("Sum : %d\n",sum);
    }
    MPI_Finalize();
    return 0;
}
