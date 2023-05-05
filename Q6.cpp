#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define N 1000
double a[N][N], b[N][N], c[N][N];

int main(int argc, char **argv)
{
    int i, j, k, nthreads, tid;
    double start, stop;

    // Initialize matrices
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            a[i][j] = 1.0;
            b[i][j] = 2.0;
            c[i][j] = 0.0;
        }
    }

    start = omp_get_wtime();

// Parallel matrix multiplication
#pragma omp parallel shared(a, b, c, nthreads) private(i, j, k, tid)
    {
        tid = omp_get_thread_num();
        if (tid == 0)
        {
            nthreads = omp_get_num_threads();
            printf("Starting matrix multiplication with %d threads\n", nthreads);
        }
#pragma omp for schedule(static)
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                for (k = 0; k < N; k++)
                {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }

    stop = omp_get_wtime();

    printf("Result matrix:\n");
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            printf("%f ", c[i][j]);
        }
        printf("\n");
    }

    printf("Time taken: %f seconds\n", stop - start);

    return 0;
}
