#include "cuda_test.h"
#include <iostream>

int data[DATA_SIZE];

void generate_rand_array(int *dest, int size)
{
    srand((int)time(0));
    for(int i = 0; i<size; i++)
        dest[i] = rand()%10000;
}


__global__ static void ArraySumOriginal(int *num, double *result)
{
    double sum = 0;
    for(int i = 0; i<DATA_SIZE; i++)
        sum += num[i]*num[i];
    *result = sum;
}
__global__ static void ArraySumThreadParallelNoContinuous(int *num, double *result)
{
    int tid = threadIdx.x;
    int size = DATA_SIZE / THREAD_NUM;
    double sum = 0;

    if(tid != THREAD_NUM - 1)
    {
        for(int i = size * tid; i < size * (tid + 1); i++)
            sum += num[i]*num[i];
    }
    else
        for(int i = size * tid; i < DATA_SIZE; i++)
            sum += num[i]*num[i];
    result[tid] = sum;
}
__global__ static void ArraySumThreadParallelContinuous(int *num, double *result)
{
    int tid = threadIdx.x;
    double sum = 0;

    for(int i = tid; i<DATA_SIZE; i+= THREAD_NUM)
        sum += num[i]*num[i];

    result[tid] = sum;
}
__global__ static void ArraySumThreadBlockParallel(int *num, double *result)
{
    int tid = blockIdx.x * THREAD_NUM + threadIdx.x;
    double sum = 0;

    for(int i = tid; i<DATA_SIZE; i+= BLOCK_NUM * THREAD_NUM)
        sum += num[i]*num[i];

    result[tid] = sum;
}
__global__ static void ArraySumThreadBlockParallelSharedMemory(int *num, double *result)
{
    extern __shared__ double shared[];
    int tid = threadIdx.x, bid = blockIdx.x;
    shared[tid] = 0;

    for(int i = bid * THREAD_NUM + tid; i<DATA_SIZE; i+= BLOCK_NUM * THREAD_NUM)
        shared[tid] += num[i]*num[i];

    __syncthreads();

    if(tid == 0)
    {
        for(int i = 1; i<THREAD_NUM; i++)
            shared[0] += shared[i];
        result[bid] = shared[0];
    }
}

__global__ static void ArraySumThreadBlockParallelSharedMemoryTreePlus(int *num, double *result)
{
    extern __shared__ double shared[];
    int tid = threadIdx.x, bid = blockIdx.x;
    shared[tid] = 0;

    for(int i = bid * THREAD_NUM + tid; i<DATA_SIZE; i+= BLOCK_NUM * THREAD_NUM)
        shared[tid] += num[i]*num[i];

    __syncthreads();

    int offset = 1, mask = 1;
    while (offset < THREAD_NUM)
    {
        if ((tid & mask) == 0)
        {
            shared[tid] += shared[tid + offset];
        }

        offset += offset;
        mask = offset + mask;
        __syncthreads();

    }

    if (tid == 0)
    {
        result[bid] = shared[0];
    }

}

void test_array_sum_cpu()
{
    double sum = 0;
    clock_t start = clock();
    for(int i = 0; i<DATA_SIZE; i++)
    {
        sum += data[i]*data[i];
    }
    clock_t duration = clock() - start;
    printf("Mode 0: CPU sum: %lf  CPU Calculation time: %lf\n", sum, (double)duration / CLOCKS_PER_SEC);

}

void test_array_sum_gpu_original()
{

    int *gpu_data;
    double sum, *gpu_sum;
    clock_t start, finish, cal_start, cal_finish;

    start = clock();
    cudaMalloc((void **)&gpu_data, sizeof(int) * DATA_SIZE);
    cudaMalloc((void **)&gpu_sum, sizeof(double));
    cudaMemcpy(gpu_data, data, sizeof(int)*DATA_SIZE, cudaMemcpyHostToDevice);

    cal_start = clock();
    //GPU实现：无任何优化
    ArraySumOriginal<<<1, 1, 0>>>(gpu_data, gpu_sum);
    cudaDeviceSynchronize();
    cal_finish = clock();

    cudaMemcpy(&sum, gpu_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(gpu_data);
    cudaFree(gpu_sum);
    finish = clock();

    printf("Mode 1: GPU sum: %lf  GPU caltulation time: %lf Total time: %lf\n", sum, 1.0 * (cal_finish - cal_start) / CLOCKS_PER_SEC , 1.0 * (finish - start) / CLOCKS_PER_SEC);
}
void test_array_sum_gpu_threads_parallel_no_continuous()
{
    int *gpu_data;
    double total_sum = 0, sum[THREAD_NUM], *gpu_sum;
    clock_t start, finish, cal_start, cal_finish;

    start = clock();
    cudaMalloc((void **)&gpu_data, sizeof(int) * DATA_SIZE);
    cudaMalloc((void **)&gpu_sum, sizeof(double) * THREAD_NUM);
    cudaMemcpy(gpu_data, data, sizeof(int)*DATA_SIZE, cudaMemcpyHostToDevice);

    cal_start = clock();
    //GPU实现：多线程，内存不连续
    ArraySumThreadParallelNoContinuous<<<1, THREAD_NUM, 0>>>(gpu_data, gpu_sum);
    cudaDeviceSynchronize();
    cal_finish = clock();

    cudaMemcpy(&sum, gpu_sum, sizeof(double) * THREAD_NUM, cudaMemcpyDeviceToHost);
    for(int i = 0; i<THREAD_NUM; i++)
        total_sum += sum[i];
    cudaFree(gpu_data);
    cudaFree(gpu_sum);
    finish = clock();

    printf("Mode 2: GPU sum: %lf  GPU caltulation time: %lf Total time: %lf\n", total_sum, 1.0 * (cal_finish - cal_start) / CLOCKS_PER_SEC, 1.0 * (finish - start) / CLOCKS_PER_SEC);


}
void test_array_sum_gpu_threads_parallel_continuous()
{
    int *gpu_data;
    double total_sum = 0, sum[THREAD_NUM], *gpu_sum;
    clock_t start, finish, cal_start, cal_finish;

    start = clock();
    cudaMalloc((void **)&gpu_data, sizeof(int) * DATA_SIZE);
    cudaMalloc((void **)&gpu_sum, sizeof(double) * THREAD_NUM);
    cudaMemcpy(gpu_data, data, sizeof(int)*DATA_SIZE, cudaMemcpyHostToDevice);

    cal_start = clock();
    //GPU实现：多线程连续内存访问
    ArraySumThreadParallelContinuous<<<1, THREAD_NUM, 0>>>(gpu_data, gpu_sum);
    cudaDeviceSynchronize();
    cal_finish = clock();
    cudaMemcpy(&sum, gpu_sum, sizeof(double) * THREAD_NUM, cudaMemcpyDeviceToHost);
    for(int i = 0; i<THREAD_NUM; i++)
        total_sum += sum[i];
    cudaFree(gpu_data);
    cudaFree(gpu_sum);
    finish = clock();
    printf("Mode 3: GPU sum: %lf  GPU caltulation time: %lf Total time: %lf\n", total_sum, 1.0 * (cal_finish - cal_start) / CLOCKS_PER_SEC, 1.0 * (finish - start) / CLOCKS_PER_SEC);

}
void test_array_sum_gpu_thread_block_parallel()
{
    int *gpu_data;
    double total_sum = 0, sum[THREAD_NUM * BLOCK_NUM], *gpu_sum;
    clock_t start, finish, cal_start, cal_finish;

    start = clock();
    cudaMalloc((void **)&gpu_data, sizeof(int) * DATA_SIZE);
    cudaMalloc((void **)&gpu_sum, sizeof(double) * THREAD_NUM * BLOCK_NUM);
    cudaMemcpy(gpu_data, data, sizeof(int)*DATA_SIZE, cudaMemcpyHostToDevice);

    cal_start = clock();
    //GPU实现：多线程连续内存访问
    ArraySumThreadBlockParallel<<<BLOCK_NUM, THREAD_NUM, 0>>>(gpu_data, gpu_sum);
    cudaDeviceSynchronize();
    cal_finish = clock();
    cudaMemcpy(&sum, gpu_sum, sizeof(double) * THREAD_NUM * BLOCK_NUM, cudaMemcpyDeviceToHost);
    for(int i = 0; i<THREAD_NUM*BLOCK_NUM; i++)
        total_sum += sum[i];
    cudaFree(gpu_data);
    cudaFree(gpu_sum);
    finish = clock();
    printf("Mode 4: GPU sum: %lf  GPU caltulation time: %lf Total time: %lf\n", total_sum, 1.0 * (cal_finish - cal_start) / CLOCKS_PER_SEC, 1.0 * (finish - start) / CLOCKS_PER_SEC);
}
void test_array_sum_gpu_thread_block_parallel_shared_memory()
{
    int *gpu_data;
    double total_sum = 0, sum[BLOCK_NUM], *gpu_sum;
    clock_t start, finish, cal_start, cal_finish;

    start = clock();
    cudaMalloc((void **)&gpu_data, sizeof(int) * DATA_SIZE);
    cudaMalloc((void **)&gpu_sum, sizeof(double) * BLOCK_NUM);
    cudaMemcpy(gpu_data, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

    cal_start = clock();
    //GPU实现：多进程块多线程并行访问 +　共享内存
    ArraySumThreadBlockParallelSharedMemory<<<BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(double)>>>(gpu_data, gpu_sum);
    cudaDeviceSynchronize();
    cal_finish = clock();
    cudaMemcpy(&sum, gpu_sum, sizeof(double) * BLOCK_NUM, cudaMemcpyDeviceToHost);
    for(int i = 0; i<BLOCK_NUM; i++)
        total_sum += sum[i];
    cudaFree(gpu_data);
    cudaFree(gpu_sum);
    finish = clock();
    printf("Mode 5: GPU sum: %lf  GPU caltulation time: %lf Total time: %lf\n", total_sum, 1.0 * (cal_finish - cal_start) / CLOCKS_PER_SEC, 1.0 * (finish - start) / CLOCKS_PER_SEC);

}
void test_array_sum_gpu_thread_block_parallel_shared_memory_tree_plus()
{
    int *gpu_data;
    double total_sum = 0, sum[BLOCK_NUM], *gpu_sum;
    clock_t start, finish, cal_start, cal_finish;

    start = clock();
    cudaMalloc((void **)&gpu_data, sizeof(int) * DATA_SIZE);
    cudaMalloc((void **)&gpu_sum, sizeof(double) * BLOCK_NUM);
    cudaMemcpy(gpu_data, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

    cal_start = clock();
    //GPU实现：多进程块多线程并行访问 + 树状加法
    ArraySumThreadBlockParallelSharedMemoryTreePlus<<<BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(double)>>>(gpu_data, gpu_sum);
    cudaDeviceSynchronize();
    cal_finish = clock();
    cudaMemcpy(&sum, gpu_sum, sizeof(double) * BLOCK_NUM, cudaMemcpyDeviceToHost);
    for(int i = 0; i<BLOCK_NUM; i++)
        total_sum += sum[i];
    cudaFree(gpu_data);
    cudaFree(gpu_sum);
    finish = clock();
    printf("Mode 6: GPU sum: %lf  GPU caltulation time: %lf Total time: %lf\n", total_sum, 1.0 * (cal_finish - cal_start) / CLOCKS_PER_SEC, 1.0 * (finish - start) / CLOCKS_PER_SEC);

}



void test_array_sum()
{

    printf("Test Array's Suqare Sum(%d integers from 0 to 10000): \n", DATA_SIZE);
    generate_rand_array(data, DATA_SIZE);
    test_array_sum_cpu();
    test_array_sum_gpu_original();
    test_array_sum_gpu_threads_parallel_no_continuous();
    test_array_sum_gpu_threads_parallel_continuous();
    test_array_sum_gpu_thread_block_parallel();
    test_array_sum_gpu_thread_block_parallel_shared_memory();
    test_array_sum_gpu_thread_block_parallel_shared_memory_tree_plus();
}