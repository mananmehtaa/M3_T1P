#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <CL/cl.h>

#define N 1024
#define BLOCK_SIZE 32

// Kernel source code
const char *kernel_source = "_kernel void matrix_mul(_global float *A, __global float *B, __global float *C, const int n) {\n"
                            "    int i = get_global_id(0);\n"
                            "    int j = get_global_id(1);\n"
                            "    float sum = 0.0f;\n"
                            "    for (int k = 0; k < n; k++) {\n"
                            "        sum += A[i * n + k] * B[k * n + j];\n"
                            "    }\n"
                            "    C[i * n + j] = sum;\n"
                            "}\n";

int main(int argc, char **argv)
{
    int i, j, k, rank, size, numBlocks;
    float *A, *B, *C;
    float *A_block, *B_block, *C_block;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem d_A, d_B, d_C;
    size_t global_size[2], local_size[2];
    int err;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check number of processes
    if (size > N / BLOCK_SIZE)
    {
        fprintf(stderr, "Number of processes must not exceed %d\n", N / BLOCK_SIZE);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize OpenCL context, queue, and kernel
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "matrix_mul", &err);

    // Allocate memory on host
    A = (float *)malloc(N * N * sizeof(float));
    B = (float *)malloc(N * N * sizeof(float));
    C = (float *)calloc(N * N, sizeof(float));

    // Initialize A and B matrices
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i * N + j] = i + j;
            B[i * N + j] = i * j;
        }
    }

    // Allocate memory on device
    A_block = (float *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
    B_block = (float *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
    C_block = (float *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
    d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, BLOCK_SIZE * N * sizeof(float), NULL, &err);
    d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, BLOCK_SIZE * N * sizeof(float), NULL, &err);
    d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, BLOCK_SIZE * N * sizeof(float), NULL, &err);

    // Copy data from host to device
    for (i = 0; i < N / BLOCK_SIZE; i++)
    {
        for (j = 0; j < BLOCK_SIZE; j++)
        {
            memcpy(&A_block[j * BLOCK_SIZE], &A[(i * BLOCK_SIZE + j) * N], BLOCK_SIZE * sizeof(float));
            memcpy(&B_block[j * BLOCK_SIZE], &B[(i * BLOCK_SIZE + j) * N], BLOCK_SIZE * sizeof(float));
        }
        clEnqueueWriteBuffer(queue, d_A, CL_TRUE, i * BLOCK_SIZE * N * sizeof(float), BLOCK_SIZE * N * sizeof(float), A_block, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, d_B, CL_TRUE, i * BLOCK_SIZE * N * sizeof(float), BLOCK_SIZE * N * sizeof(float), B_block, 0, NULL, NULL);
    }

    // Launch kernel
    for (i = 0; i < N / BLOCK_SIZE; i++)
    {
        // Set kernel arguments
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
        clSetKernelArg(kernel, 3, sizeof(int), &N);
        clSetKernelArg(kernel, 4, sizeof(int), &BLOCK_SIZE);
        clSetKernelArg(kernel, 5, sizeof(int), &i);

        // Set workgroup size
        global_size[0] = N;
        global_size[1] = N;
        local_size[0] = BLOCK_SIZE;
        local_size[1] = BLOCK_SIZE;

        // Execute kernel
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    }

    // Copy data from device to host
    for (i = 0; i < N / BLOCK_SIZE; i++)
    {
        clEnqueueReadBuffer(queue, d_C, CL_TRUE, i * BLOCK_SIZE * N * sizeof(float), BLOCK_SIZE * N * sizeof(float), &C[(i * BLOCK_SIZE) * N], 0, NULL, NULL);
    }

    // Free memory
    free(A);
    free(B);
    free(C);
    free(A_block);
    free(B_block);
    free(C_block);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    MPI_Finalize();

    return 0;
}
