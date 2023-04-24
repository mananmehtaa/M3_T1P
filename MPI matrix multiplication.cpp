// Include standard input-output header file
#include <stdio.h>
// Include MPI header file for MPI function declarations
#include <mpi.h>

// Define a macro SIZE as 4
#define SIZE 4

// Define main function with command line arguments
int main(int argc, char *argv[])
{
    // Declare variables i, j, k, rank and size as integers, and A, B, and C as 2-D double arrays
    int i, j, k, rank, size;
    double A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    // Get rank of the process and store in rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Get total number of processes and store in size
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Check if the total number of processes is equal to SIZE
    if (size != SIZE)
    {
        // If not, print an error message to standard error stream and abort the program
        fprintf(stderr, "Must run with %d processes\n", SIZE);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize A, B and C matrices
    for (i = 0; i < SIZE; i++)
    {
        for (j = 0; j < SIZE; j++)
        {
            A[i][j] = i + j;
            B[i][j] = i * j;
            C[i][j] = 0;
        }
    }

    // Wait until all processes reach this point before continuing
    MPI_Barrier(MPI_COMM_WORLD);

    // Calculate C matrix elements for each process
    for (i = rank; i < SIZE; i += size)
    {
        for (j = 0; j < SIZE; j++)
        {
            for (k = 0; k < SIZE; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // Reduce the partial C matrices calculated by each process into a single C matrix
    MPI_Reduce(C, C, SIZE * SIZE, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // If this is the root process (rank 0), print the resulting C matrix
    if (rank == 0)
    {
        printf("Matrix C:\n");
        for (i = 0; i < SIZE; i++)
        {
            for (j = 0; j < SIZE; j++)
            {
                printf("%f ", C[i][j]);
            }
            printf("\n");
        }
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}