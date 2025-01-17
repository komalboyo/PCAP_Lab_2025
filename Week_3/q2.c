// Read an int M and N x M elements into array in root, root sends M elements to each process , each process 
// finds avg of M elements and sends avg to root, root collects all and finds total avg

#include<stdio.h>
#include<mpi.h>
#include<unistd.h>

int main(int argc, char *argv[]){
	int rank, size, N, M;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	N = size;
	MPI_Status status;

	if(rank==0){
        printf("Enter the value of M : ");
        scanf("%d", &M);
    }
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("P%d rcvd M = %d\n", rank, M);
    int arr_size = N * M;
    int A[arr_size];
    int rcvbuf[M];
    float averages[N];

    if (rank==0){  	
    	sleep(1);
        printf("Enter %d array elements : ", arr_size);
        for(int i=0;i<arr_size;i++)
            scanf("%d",&A[i]);
    }

    MPI_Scatter(A,M,MPI_INT,&rcvbuf,M,MPI_INT,0,MPI_COMM_WORLD);

    float avg=0.0;
    for (int i=0;i<M;i++){
    	avg += rcvbuf[i];
    }
    avg = avg/M;

	MPI_Gather(&avg,1,MPI_INT,averages,1,MPI_INT,0,MPI_COMM_WORLD);

	if(rank==0){
    	sleep(1);	// cuz printing takes time
        printf("Gathered results : \n");
        avg = 0.0;
        for(int i=0;i<N;i++){
            printf("%f \n",averages[i]);
            avg += averages[i];
        }
        avg = avg/N;
        printf("Average : %f\n",avg);
    }

    MPI_Finalize();
    return 0;
}
