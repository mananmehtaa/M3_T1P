#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <time.h>

#define N 1000

int main()
{
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem mem_a, mem_b, mem_c;
    double start, end;
    int i, j, k;
    float *a = (float *)malloc(N * N * sizeof(float));
    float *b = (float *)malloc(N * N * sizeof(float));
    float *c = (float *)calloc(N * N, sizeof(float));

    // Initialize matrices a and b
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            a[i * N + j] = 1.0;
            b[i * N + j] = 2.0;
        }
    }

    // Get platform and device information
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create an OpenCL context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating context\n");
        exit(1);
    }

    // Create a command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating command queue\n");
        exit(1);
    }

    // Create memory buffers on the device for each matrix
    mem_a = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating buffer for matrix a\n");
        exit(1);
    }

    mem_b = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating buffer for matrix b\n");
        exit(1);
    }

    mem_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * N * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating buffer for matrix c\n");
        exit(1);
    }

    // Write matrices a and b to the device
    clEnqueueWriteBuffer(queue, mem_a, CL_TRUE, 0, N * N * sizeof(float), a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, mem_b, CL_TRUE, 0, N * N * sizeof(float), b, 0, NULL, NULL);

    // Create a program from the kernel source
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating program\n");
        exit(1);
    }

    // Build the program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error building program\n");
        exit(1);
    }

    // Create the OpenCL kernel
    kernel = clCreateKernel(program, "matrix_mul", &err);
    if (err != CLSUCCESS)
    {
        printf("Error creating kernel\n");
        exit(1);
    }

    // Set the arguments of the kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mem_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&mem_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&mem_c);
    err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&N);
    if (err != CL_SUCCESS)
    {
        printf("Error setting kernel arguments\n");
        exit(1);
    }

    // Execute the OpenCL kernel on the input data
    size_t global_size[2] = {N, N};
    size_t local_size[2] = {16, 16};
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);

    // Read the result from the device
    clEnqueueReadBuffer(queue, mem_c, CL_TRUE, 0, N * N * sizeof(float), c, 0, NULL, NULL);

    // Verify the result
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            float sum = 0.0;
            for (k = 0; k < N; k++)
            {
                sum += a[i * N + k] * b[k * N + j];
            }
            if (fabs(c[i * N + j] - sum) > 1e-5)
            {
                printf("Error: incorrect result at position (%d, %d)\n", i, j);
                exit(1);
            }
        }
    }

    printf("Matrix multiplication completed successfully\n");

    // Clean up
    clReleaseMemObject(mem_a);
    clReleaseMemObject(mem_b);
    clReleaseMemObject(mem_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(a);
    free(b);
    free(c);

    return 0;
}
