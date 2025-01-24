// Read a word of length N, using N processes including root, get output word st. each consecutive letter is repeated in progression number of times

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ErrorHandler(int err_code) {
    if (err_code != MPI_SUCCESS) {
        char error_string[BUFSIZ];
        int length_err_string, err_class;
        MPI_Error_class(err_code, &err_class);
        MPI_Error_string(err_code, error_string, &length_err_string);
        printf("Error: %d %s\n", err_class, error_string);
    }
}

int main(int argc, char *argv[]) {
    int rank, size, err_code;
    char str[100];
    char resultant[1000]; 

    MPI_Init(&argc, &argv);
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    err_code = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    err_code = MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        ErrorHandler(err_code);
        printf("Enter the string: \n");
        scanf("%[^\n]c", str);
        if (size != strlen(str)) {
            printf("Error: This program requires the number of processes = length of the string.\n");
            exit(1);
        }
    }

    char rcvbuf[2];
    err_code = MPI_Scatter(str, 1, MPI_CHAR, rcvbuf, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    ErrorHandler(err_code);

    char modified_str[100] = {0}; // Buffer to store the repeated character string

    for (int i = 0; i < rank + 1; i++) {
        modified_str[i] = rcvbuf[0];
    }
    modified_str[rank + 1] = '\0';

    char temp_result[1000];
    err_code = MPI_Gather(modified_str, 100, MPI_CHAR, temp_result, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
    ErrorHandler(err_code);

    if (rank == 0) {
        resultant[0] = '\0';
        for (int i = 0; i < size; i++) {
            strcat(resultant, &temp_result[i * 100]);  // Append each process's result to the final string
        }
        printf("The final result is: %s\n", resultant);
    }

    MPI_Finalize();
    exit(0);
}
