#include <stdio.h>
#include <time.h>

#define N 1000

int main()
{
    double a[N][N], b[N][N], c[N][N];
    int i, j, k;

    // Initialize matrices
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            a[i][j] = 1.0;
            b[i][j] = 2.0;
        }
    }

    // Start timer
    clock_t start = clock();

    // Multiply matrices
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

    // Stop timer
    clock_t stop = clock();

    // Calculate elapsed time
    double elapsed_time = (double)(stop - start) / CLOCKS_PER_SEC;

    // Print result and elapsed time
    printf("Result matrix:\n");
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            printf("%.2f ", c[i][j]);
        }
        printf("\n");
    }
    printf("Elapsed time: %.6f seconds\n", elapsed_time);

    return 0;
}
