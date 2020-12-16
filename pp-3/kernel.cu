
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#define BLOCK_SIZE 32

__global__ void gpu_square_matrix_mult(int* d_a, int* d_b, int* d_result, int n)
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub)
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if (idx >= n * n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if (idx >= n * n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}

__constant__ int * c_a[1000], * c_b[1000], * c_c[1000];
__global__ void gpu_square_matrix_mult_cnst(int* d_a, int* d_b, int* d_result)
{
    int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub)
    {
        idx = row * 1000 + sub * BLOCK_SIZE + threadIdx.x;
        if (idx >= 1000 * 1000)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * 1000 + col;
        if (idx >= 1000 * 1000)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
    }
    if (row < 1000 && col < 1000)
    {
        d_result[row * 1000 + col] = tmp;
    }
}

__global__ void gpu_square_matrix_mult_glbl(int* d_a, int* d_b, int* d_result, int n)
{
    int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub)
    {
        idx = row * 1000 + sub * BLOCK_SIZE + threadIdx.x;
        if (idx >= 1000 * 1000)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * 1000 + col;
        if (idx >= 1000 * 1000)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
    }
    if (row < 1000 && col < 1000)
    {
        d_result[row * 1000 + col] = tmp;
    }
}

__global__ void gpu_matrix_transpose(int* mat_in, int* mat_out, unsigned int rows, unsigned int cols)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows)
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}

void cpu_matrix_mult(int* h_a, int* h_b, int* h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h)
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

int dtn(int n, int min_n)
{
    int max_tn = n / min_n;
    const int g_ncore = omp_get_num_procs();
    int tn = max_tn > g_ncore ? g_ncore : max_tn;
    if (tn < 1)
    {
        tn = 1;
    }
    return tn;
}

void omp_mm(int* a, int row_a, int col_a, int* b, int row_b, int col_b, int* c)
{
    if (col_a != row_b)
    {
        return;
    }
    int i, j, k;
    int index;
    int border = row_a * col_b;
    double sum = 0;
    i = 0;
    j = 0;

#pragma omp parallel for private(i,j,k) num_threads(dtn(border, 1))
    for (index = 0; index < border; index++)
    {
        i = index / col_b; j = index % col_b;
        int row_i = i * col_a;
        int row_c = i * col_b;
        c[row_c + j] = 0;
        for (k = 0; k < row_b; k++)
        {
            c[row_c + j] += a[row_i + k] * b[k * col_b + j];
            sum = sum + c[row_c + j];
        }
    }
}

int main(int argc, char const* argv[])
{
    int m, n, k;
    /* Fixed seed for illustration */
    srand(3333);
    printf("please type in n\n");
    scanf("%d", &n);
    m = n;
    k = n;

    // allocate memory in host RAM, h_cc is used to store CPU result
    int* h_a, * h_b, * h_c, * h_cc;
    int* c_a, * c_b, * c_c;
    cudaMallocHost((void**)&h_a, sizeof(int) * m * n);
    cudaMallocHost((void**)&h_b, sizeof(int) * n * k);
    cudaMallocHost((void**)&h_c, sizeof(int) * m * k);
    cudaMallocHost((void**)&h_cc, sizeof(int) * m * k);
    cudaMallocHost((void**)&c_a, sizeof(int) * m * n);
    cudaMallocHost((void**)&c_b, sizeof(int) * n * k);
    cudaMallocHost((void**)&c_c, sizeof(int) * m * k);

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024;
            c_a[i * n + j] = rand() % 1024;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 10;
            c_b[i * n + j] = rand() % 1024;
        }
    }

    float gpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    // Allocate memory space on the device 
    int* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, sizeof(int) * m * n);
    cudaMalloc((void**)&d_b, sizeof(int) * n * k);
    cudaMalloc((void**)&d_c, sizeof(int) * m * k);

    // copy matrix A and B and sum from host to device memory
    cudaEventRecord(start, 0);
    cudaMemcpy(d_a, h_a, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * n * k, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time of copying data from host to device memory %f ms\n", gpu_elapsed_time_ms);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    cudaEventRecord(start, 0);
    gpu_square_matrix_mult << <dimGrid, dimBlock >> > (d_a, d_b, d_c, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %d on GPU shared: %f ms.\n", n, gpu_elapsed_time_ms);


    // Transefr results from device to host 
    cudaEventRecord(start, 0);
    cudaMemcpy(h_c, d_c, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on transfer results from device to host shared: %f ms.\n\n", n, gpu_elapsed_time_ms);


    cudaEventRecord(start, 0);
    cudaMemcpy(c_a, h_a, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(c_b, h_b, sizeof(int) * n * k, cudaMemcpyHostToDevice);
    cudaMemcpy(c_c, h_c, sizeof(int) * n * k, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time of copying data from host to device memory constant %f ms\n", gpu_elapsed_time_ms);

    cudaEventRecord(start, 0);
    gpu_square_matrix_mult_cnst << <dimGrid, dimBlock >> > (c_a, c_b, c_c);
    cudaEventRecord(stop, 0);
    cudaThreadSynchronize();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %d on GPU constant: %f ms.\n", n, gpu_elapsed_time_ms);

    cudaEventRecord(start, 0);
    cudaMemcpy(h_c, d_c, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on transfer results from device to host constant: %f ms.\n\n", n, gpu_elapsed_time_ms);


    cudaEventRecord(start, 0);
    cudaMalloc((void**)&d_a, sizeof(int) * m * n);
    cudaMalloc((void**)&d_b, sizeof(int) * n * k);
    cudaMalloc((void**)&d_c, sizeof(int) * m * k);
    cudaMemcpy(d_a, h_a, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * n * k, cudaMemcpyHostToDevice);  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time of copying data from host to device memory global %f ms\n", gpu_elapsed_time_ms);

    cudaEventRecord(start, 0);
    gpu_square_matrix_mult_glbl << <dimGrid, dimBlock >> > (d_a, d_b, d_c, n);
    cudaEventRecord(stop, 0);
    cudaThreadSynchronize();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of % d on GPU global: % f ms.\n", n, gpu_elapsed_time_ms);

    cudaEventRecord(start, 0);
    cudaMemcpy(h_c, d_c, sizeof(int)* m* k, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on transfer results from device to host global: %f ms.\n\n", n, gpu_elapsed_time_ms);
    
    // start the CPU version
    double s = omp_get_wtime();
    omp_mm(h_a, m, n, h_b, n, k, h_cc);
    double e = omp_get_wtime();
    printf("\nTime elapsed on matrix multiplication of %d on CPU: %f s.\n\n", n, e - s);

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    return 0;
}
