// Include necessary header files
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

// Define constants N and BLOCK_SIZE
#define N 1000
#define BLOCK_SIZE 100

// Main function starts
int main(int argc, char **argv)
{
    // Declare variables
    int i, j, k, rank, size, num_threads;
    double **A, **B, **C, *A_block, *B_block, *C_block;
    double start_time, end_time, total_time;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank of the current process and the total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Enable nested parallelism
    omp_set_nested(1);

    // Check if the number of processes does not exceed N/BLOCK_SIZE
    if (size > N / BLOCK_SIZE)
    {
        fprintf(stderr, "Number of processes must not exceed %d\n", N / BLOCK_SIZE);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate memory for matrices A, B, C, and their blocks
    A = (double **)malloc(N * sizeof(double *));
    B = (double **)malloc(N * sizeof(double *));
    C = (double **)malloc(N * sizeof(double *));
    A_block = (double *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    B_block = (double *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    C_block = (double *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

    // Allocate memory for the rows of matrices A, B, and C
    for (i = 0; i < N; i++)
    {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)malloc(N * sizeof(double));
        C[i] = (double *)malloc(N * sizeof(double));
    }

    // Initialize matrices A and B, and set all values of C to 0
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i][j] = i + j;
            B[i][j] = i * j;
            C[i][j] = 0;
        }
    }

    // Start timer
    start_time = MPI_Wtime();

    // Perform matrix multiplication in parallel
    for (i = rank * BLOCK_SIZE; i < (rank + 1) * BLOCK_SIZE; i += num_threads)
    {
        // Get the maximum number of threads
        num_threads = omp_get_max_threads();

// Perform block matrix multiplication
#pragma omp parallel for private(j, k) shared(A, B, C, i, A_block, B_block, C_block)
        for (j = 0; j < BLOCK_SIZE; j++)
        {
            for (k = 0; k < N; k += BLOCK_SIZE)
            {
                A_block[j * BLOCK_SIZE + k % BLOCK_SIZE] = A[i + j][k];
                B_block[j * BLOCK_SIZE + k % BLOCK_SIZE] = B[k][i + j];
            }
        }

// Perform matrix multiplication on the blocks
#pragma omp parallel for private(j, k) shared(A_block, B_block, C_block)
        for (j = 0; j < BLOCK_SIZE; j++)
        {
            for (k = 0; k < BLOCK_SIZE; k++)
            {
                / Initialize a variable "sum" and set it to 0 double sum = 0;

                // Initialize a variable "l" for the loop
                int l;

                // Loop through the rows of A_block and columns of B_block to compute the matrix multiplication for the given block
                for (l = 0; l < N; l++)
                {
                    sum += A_block[j * BLOCK_SIZE + l % BLOCK_SIZE] * B_block[l % BLOCK_SIZE * BLOCK_SIZE + k];
                }

                // Store the computed value in the appropriate position in the result matrix block C_block
                C_block[j * BLOCK_SIZE + k] = sum;

// Parallelize the outer two loops for the matrix multiplication using OpenMP
#pragma omp parallel for private(j, k) shared(C, C_block, i)
                for (j = 0; j < BLOCK_SIZE; j++)
                {
                    for (k = 0; k < N; k += BLOCK_SIZE)
                    {
                        int x, y;
                        // Loop through the rows and columns of the current block to add the values to the corresponding position in the result matrix C
                        for (x = i + j; x < i + j + BLOCK_SIZE; x++)
                        {
                            for (y = k; y < k + BLOCK_SIZE; y++)
                            {
                                C[x][y] += C_block[(x - i - j) * BLOCK_SIZE + (y - k)];
                            }
                        }
                    }

                    // Wait for all processes to reach this point before continuing
                    MPI_Barrier(MPI_COMM_WORLD);

                    // Compute the time taken for the matrix multiplication
                    end_time = MPI_Wtime();
                    total_time = end_time - start_time;

                    // If the process is the root process, print the total time taken for the computation
                    if (rank == 0)
                    {
                        printf("Total time: %f seconds\n", total_time);
                    }

                    // Free the dynamically allocated memory for the matrices A, B, and C, as well as the memory for the A_block, B_block, and C_block
                    free(A_block);
                    free(B_block);
                    free(C_block);
                    for (i = 0; i < N; i++)
                    {
                        free(A[i]);
                        free(B[i]);
                        free(C[i]);
                    }
                    free(A);
                    free(B);
                    free(C);

                    // Finalize the MPI environment
                    MPI_Finalize();
                }
            }
        }
    }
}
