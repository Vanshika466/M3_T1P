#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <CL/cl.h>
#include <time.h>
#include <string.h>

#define N 512
#define LOCAL_SIZE 16
#define VERIFY_ELEMENTS 4

const char *kernel_source =
"__kernel void matrix_mult(__global float *A, __global float *B, __global float>
"    int i = get_global_id(0);\n"
"    int j = get_global_id(1);\n"
"    float sum = 0.0f;\n"
"    for (int k = 0; k < N; k++) {\n"
"        sum += A[i*N + k] * B[k*N + j];\n"
"    }\n"
"    C[i*N + j] = sum;\n"
"}\n";

void print_device_info(cl_device_id device, int rank) {
    char device_name[128], vendor[128];
    cl_device_type type;
    cl_uint freq, cores;
    size_t wg_size;
   
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, N>
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq,>
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cores), &cores,>
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wg_size), &wg>
   
    printf("Rank %d: %s (%s)\n", rank, device_name, vendor);
    printf("Rank %d: Type: %s\n", rank,
          (type & CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU");
printf("Rank %d: Cores: %u @ %u MHz\n", rank, cores, freq);
    printf("Rank %d: Max WG Size: %zu\n", rank, wg_size);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
   
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0 && rank == 0) {
        fprintf(stderr, "Matrix size must be divisible by process count\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Timing
double start = MPI_Wtime();
    int rows_per_process = N / size;

    // Host matrices
    float *A = NULL, *B = NULL, *C = NULL;
    float *local_A = malloc(rows_per_process * N * sizeof(float));
    float *local_C = malloc(rows_per_process * N * sizeof(float));
    float *B_global = malloc(N * N * sizeof(float));

    if (rank == 0) {
        A = malloc(N * N * sizeof(float));
        C = malloc(N * N * sizeof(float));

        srand(time(NULL));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i*N + j] = (i == j) ? 1.0f : 0.0f;  // Identity matrix
                B_global[i*N + j] = (float)(i + 1);    // Sequential values
 }
        }
    }

    MPI_Scatter(A, rows_per_process*N, MPI_FLOAT, local_A, rows_per_process*N, >
    MPI_Bcast(B_global, N*N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // OpenCL setup
    cl_platform_id platform;
    cl_device_id device = NULL;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem A_dev, B_dev, C_dev;
cl_int err;
    cl_uint num_platforms, num_devices;

    // Get platform
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "Rank %d: No OpenCL platforms found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Get all GPU devices
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    cl_device_id *gpus = NULL;
    if (err == CL_SUCCESS && num_devices > 0) {
        gpus = malloc(num_devices * sizeof(cl_device_id));
 clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, gpus, NULL);

        // Try to find Intel GPU
        for (int i = 0; i < num_devices; i++) {
            char vendor[128];
            clGetDeviceInfo(gpus[i], CL_DEVICE_VENDOR, sizeof(vendor), vendor, >
            if (strstr(vendor, "Intel")) {
                device = gpus[i];
                break;
            }
        }

        // Fallback to first GPU if no Intel found
        if (!device && num_devices > 0) device = gpus[0];
    }

    // CPU fallback
    if (!device) {
 err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Rank %d: No OpenCL devices found\n", rank);
            free(gpus);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    free(gpus);

    // Print device info
    print_device_info(device, rank);

    // Create context and queue
    cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_propertie>
    context = clCreateContext(props, 1, &device, NULL, NULL, &err);
 queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &e>

    // Create buffers
    A_dev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         rows_per_process*N*sizeof(float), local_A, &err);
    B_dev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         N*N*sizeof(float), B_global, &err);
    C_dev = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                         rows_per_process*N*sizeof(float), NULL, &err);

    // Build program
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
   
    if (err != CL_SUCCESS) {
 size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &>
        char *log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, >
        fprintf(stderr, "Rank %d: Build error:\n%s\n", rank, log);
        free(log);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Create kernel
    kernel = clCreateKernel(program, "matrix_mult", &err);
    int N_arg = N;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &A_dev);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &B_dev);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &C_dev);
    clSetKernelArg(kernel, 3, sizeof(int), &N_arg);

    // Execute kernel
 size_t global_size[2] = {rows_per_process, N};
    size_t local_size[2] = {LOCAL_SIZE, LOCAL_SIZE};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_siz>
    clFinish(queue);

    // Read results
    err = clEnqueueReadBuffer(queue, C_dev, CL_TRUE, 0,
                            rows_per_process*N*sizeof(float), local_C, 0, NULL,>

    // Gather results
    MPI_Gather(local_C, rows_per_process*N, MPI_FLOAT, C, rows_per_process*N, M>

    // Verification
    if (rank == 0) {
        printf("\nComputation time: %.4f sec\n", MPI_Wtime() - start);

        // Verify sample elements
        int test_indices[] = {0, N/4, N/2, N-1};
printf("Verification:\n");
        for (int i = 0; i < VERIFY_ELEMENTS; i++) {
            int idx = test_indices[i];
            printf("C[%d][%d] = %.2f\n", idx, idx, C[idx*N + idx]);
        }
    }

    // Cleanup
    clReleaseMemObject(A_dev);
    clReleaseMemObject(B_dev);
    clReleaseMemObject(C_dev);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
   
    free(local_A);
    free(local_C);
 free(B_global);
    if (rank == 0) {
        free(A);
        free(C);
    }
   
    MPI_Finalize();
    return 0;
}

