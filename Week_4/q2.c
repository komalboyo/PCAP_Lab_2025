// Read a 3x3 matrix, enter an element to be searched in root process. Then find the number of occurances of this element in the matrix using 3 processes

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void ErrorHandler(int err_code) {
    if(err_code != MPI_SUCCESS) {
        char error_string[BUFSIZ];
        int length_err_string, err_class;
        MPI_Error_class(err_code, &err_class);
        MPI_Error_string(err_code, error_string, &length_err_string);
        printf("Error: %d %s\n", err_class, error_string);
    }
}

void CheckProcessCount(int size, int rank) {
    if(size != 3) {
        if(rank == 0) {
            printf("Error: This program requires exactly 3 processes. You have %d processes.\n", size);
        }
        MPI_Finalize();
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    int rank, size, err_code, ele, result;
    int mat[3][3];

    MPI_Init(&argc, &argv);
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    err_code = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ErrorHandler(err_code);
    err_code = MPI_Comm_size(MPI_COMM_WORLD, &size);
    ErrorHandler(err_code);

    CheckProcessCount(size, rank);

    if(rank == 0) {
        printf("Enter the elements in 3x3 matrix:\n");
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                scanf("%d", &mat[i][j]);
        printf("Enter element to be searched: ");
        scanf("%d", &ele);
    }

    int arr[3];
    err_code = MPI_Bcast(&ele, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ErrorHandler(err_code);

    err_code = MPI_Scatter(mat, 3, MPI_INT, arr, 3, MPI_INT, 0, MPI_COMM_WORLD);
    ErrorHandler(err_code);

    int res = 0;
    for(int i = 0; i < 3; i++)
        if(arr[i] == ele)
            res++;

    err_code = MPI_Reduce(&res, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    ErrorHandler(err_code);

    if(rank == 0) {
        printf("Total number of occurrences is: %d\n", result);
    }

    MPI_Finalize();
    exit(0);
}
