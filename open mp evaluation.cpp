#include <stdio.h>
#include <omp.h>

#define N 1000

int main(int argc, char *argv[])
{
    int i, j, k;
    double start, stop;
    double a[N][N], b[N][N], c[N][N];

    // Initialize matrices a and b
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            a[i][j] = 1.0;
            b[i][j] = 2.0;
        }
    }

    start = omp_get_wtime();

// Compute matrix multiplication using OpenMP parallelization
#pragma omp parallel for shared(a, b, c) private(i, j, k)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            c[i][j] = 0.0;
            for (k = 0; k < N; k++)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    stop = omp_get_wtime();
    printf("OpenMP Program Time = %.6f seconds\n\n", stop - start);

    return 0;
}
