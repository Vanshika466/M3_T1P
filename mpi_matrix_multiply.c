#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define N 8  // Reduced size for demonstration (can change back to 1024 later)

void print_matrix(double *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%6.2f ", matrix[i*size + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
   
    int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0 && rank == 0) {
        printf("Error: Matrix size %d must be divisible by number of processes >
        MPI_Finalize();
        return 1;
    }

    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A, *local_C;
    int rows_per_process = N / size;

    // Allocate and initialize matrices
    B = (double *)malloc(N * N * sizeof(double));
   
    if (rank == 0) {
 A = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));

        // Initialize with simple values for verification
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i*N + j] = (i == j) ? 1.0 : 0.0;  // Identity matrix
                B[i*N + j] = (double)(i + 1);        // Simple pattern
            }
        }

        printf("Initial Matrix A:\n");
        print_matrix(A, N);
        printf("\nInitial Matrix B:\n");
        print_matrix(B, N);
    }

    local_A = (double *)malloc(rows_per_process * N * sizeof(double));
local_C = (double *)malloc(rows_per_process * N * sizeof(double));

    // Distribute data
    MPI_Scatter(A, rows_per_process*N, MPI_DOUBLE,
               local_A, rows_per_process*N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
   
    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i*N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                local_C[i*N + j] += local_A[i*N + k] * B[k*N + j];
            }
        }
    }
    // Gather results
    MPI_Gather(local_C, rows_per_process*N, MPI_DOUBLE,
              C, rows_per_process*N, MPI_DOUBLE,
              0, MPI_COMM_WORLD);

    // Print results
    if (rank == 0) {
        printf("\nResult Matrix C:\n");
        print_matrix(C, N);

        free(A); free(C);
    }
   
    free(B); free(local_A); free(local_C);
    MPI_Finalize();
    return 0;
}
