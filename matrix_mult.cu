#include "cuda_test.h"

double A[MATRIX_ROWS * MATRIX_COLS];
double B[MATRIX_ROWS * MATRIX_COLS];
double transB[MATRIX_COLS * MATRIX_ROWS];
double C[MATRIX_ROWS * MATRIX_ROWS];
double C_cpu[MATRIX_ROWS * MATRIX_ROWS];
void generate_rand_matrix()
{
    srand((int)time(0));
    for(int i = 0; i<MATRIX_ROWS; i++)
    {
        for(int j = 0; j<MATRIX_COLS; j++)
        {
            A[i * MATRIX_COLS + j] = (float)rand()/RAND_MAX + (float)rand()/RAND_MAX/RAND_MAX;
            B[i * MATRIX_COLS + j] = (float)rand()/RAND_MAX + (float)rand()/RAND_MAX/RAND_MAX;

        }
    }
//    for(int i = 0; i<MATRIX_ROWS * MATRIX_COLS; i++)
//        printf("%lf ", B[i]);
//    printf("\n");
}
__global__ static void TransposeMatrix(double *A, double *B)
{
    int tid = blockIdx.x * THREAD_NUM + threadIdx.x;
    for(int i = tid; i<MATRIX_ROWS*MATRIX_COLS; i += BLOCK_NUM * THREAD_NUM)
    {
        int row = tid / MATRIX_COLS;
        int col = tid % MATRIX_COLS;
        if (row < MATRIX_ROWS && col < MATRIX_COLS)
            B[row * MATRIX_COLS + col] = A[col * MATRIX_ROWS + row];
    }

}
__global__ static void MatrixMultTransposeB(double *A, double * oriB, double *B, double *C)
{
    int tid = blockIdx.x * THREAD_NUM + threadIdx.x;
    int row, col;
    double sum;
    for(int i = tid; i<MATRIX_ROWS*MATRIX_ROWS; i += BLOCK_NUM * THREAD_NUM)
    {
        sum = 0;
        row = i / MATRIX_ROWS;
        col = i % MATRIX_ROWS;
        for(int j = 0; j<MATRIX_COLS; j++)
        {
//            if(oriB[j*MATRIX_ROWS + col]!=B[col * MATRIX_COLS + j])
//                printf(" %d %d\n", row, col);
            sum += A[row*MATRIX_COLS + j] * oriB[j*MATRIX_ROWS + col];//B[col * MATRIX_COLS + j];
        }
        C[i] = sum;
    }
}
__global__ static void MatrixMultOriginal(double *A, double *B, double *C)
{
    int tid = blockIdx.x * THREAD_NUM + threadIdx.x;
    int row, col;
    double sum;
    for(int i = tid; i<MATRIX_ROWS*MATRIX_ROWS; i += BLOCK_NUM * THREAD_NUM)
    {
        sum = 0;
        row = i / MATRIX_ROWS;
        col = i % MATRIX_ROWS;
        for(int j = 0; j<MATRIX_COLS; j++)
        {
            sum += A[row*MATRIX_COLS + j] * B[j*MATRIX_ROWS + col];
        }
        C[i] = sum;
    }
}

__global__ static void MatrixMultOnebyOne(double *A, double *B, double *C)
{
    double sum;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < MATRIX_ROWS && col < MATRIX_ROWS)
    {
        sum = 0;
        for(int i = 0; i<MATRIX_COLS; i++)
        {
            sum += A[row*MATRIX_COLS + i] * B[i*MATRIX_ROWS + col];
        }
        C[row*MATRIX_ROWS + col] = sum;
    }
}
//template <int blockSize>
__global__ void MatrixMultSharedMemory(double *A, double *B, double *C)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = threadIdx.x;
    int y = threadIdx.y;

    double tsum = 0;
    for(int i = 0; i<MATRIX_COLS; i+=SHARED_BLOCK_SIZE)
    {
        __shared__ double As[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE];
        __shared__ double Bs[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE];
        if(bx * SHARED_BLOCK_SIZE + x < MATRIX_ROWS && i + y < MATRIX_COLS)
            As[x][y] = A[bx*SHARED_BLOCK_SIZE*MATRIX_COLS + x*MATRIX_COLS + i + y];
        else
            As[x][y] = 0;

        if(x + i < MATRIX_COLS && by * SHARED_BLOCK_SIZE + y < MATRIX_ROWS)
            Bs[x][y] = B[x*MATRIX_ROWS + i*MATRIX_ROWS + by*SHARED_BLOCK_SIZE + y];
        else
            Bs[x][y] = 0;

        __syncthreads();
        for(int k = 0; k<SHARED_BLOCK_SIZE; k++)
            tsum += As[x][k] * Bs[k][y];
        __syncthreads();
    }
    if(bx * SHARED_BLOCK_SIZE + x < MATRIX_ROWS && by * SHARED_BLOCK_SIZE + y < MATRIX_ROWS)
        C[bx*SHARED_BLOCK_SIZE*MATRIX_ROWS + x*MATRIX_ROWS + by*SHARED_BLOCK_SIZE + y] = tsum;


}
void test_matrix_mult_cpu()
{
    clock_t start, finish;
    memset(C_cpu, 0, sizeof(double) * MATRIX_ROWS * MATRIX_ROWS);
    start = clock();
    for(int i = 0; i<MATRIX_ROWS; i++)
    {
        for(int j = 0; j<MATRIX_ROWS; j++)
        {
            for(int k = 0; k<MATRIX_COLS; k++)
            {
                C_cpu[i*MATRIX_ROWS + j] += A[i*MATRIX_COLS + k] * B[k*MATRIX_ROWS + j];
            }
        }
    }
    finish = clock();
    printf("Mode 0: CPU Calculation time: %lf\n",finish, start, CLOCKS_PER_SEC, 1.0 * (finish - start) / CLOCKS_PER_SEC);
}
void test_matrix_mult_gpu_original()
{
    clock_t start, finish, cal_start, cal_finish;
    double *cuda_A, *cuda_B, *cuda_C;

    start = clock();
    cudaMalloc((void**)&cuda_A, sizeof(double) * MATRIX_ROWS * MATRIX_COLS);
    cudaMalloc((void**)&cuda_B, sizeof(double) * MATRIX_ROWS * MATRIX_COLS);
    cudaMalloc((void**)&cuda_C, sizeof(double) * MATRIX_ROWS * MATRIX_ROWS);
    cudaMemcpy(cuda_A, A, sizeof(double)*MATRIX_ROWS*MATRIX_COLS, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B, sizeof(double)*MATRIX_ROWS*MATRIX_COLS, cudaMemcpyHostToDevice);

    cal_start = clock();
    MatrixMultOriginal<<<BLOCK_NUM, THREAD_NUM, 0>>>(cuda_A, cuda_B, cuda_C);
    cudaMemcpy(C, cuda_C, sizeof(double)*MATRIX_ROWS*MATRIX_ROWS, cudaMemcpyDeviceToHost);
    cal_finish = clock();

    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);
    finish = clock();

    double total_err = 0;
    for(int i = 0; i<MATRIX_ROWS*MATRIX_ROWS; i++)
    {
        total_err += fabs(C[i] - C_cpu[i]);
    }
    printf("Mode 1: Total error: %lf, GPU Calculation time: %lf, Total time: %lf \n", total_err, 1.0 * (cal_finish - cal_start) / CLOCKS_PER_SEC, 1.0 * (finish - start) / CLOCKS_PER_SEC);
}
void test_matrix_mult_gpu_one_by_one()
{
    clock_t start, finish, cal_start, cal_finish;
    double *cuda_A, *cuda_B, *cuda_C;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid_size((MATRIX_ROWS+BLOCK_SIZE-1)/BLOCK_SIZE, (MATRIX_ROWS + BLOCK_SIZE -1)/BLOCK_SIZE);


    start = clock();
    cudaMalloc((void**)&cuda_A, sizeof(double) * MATRIX_ROWS * MATRIX_COLS);
    cudaMalloc((void**)&cuda_B, sizeof(double) * MATRIX_ROWS * MATRIX_COLS);
    cudaMalloc((void**)&cuda_C, sizeof(double) * MATRIX_ROWS * MATRIX_ROWS);
    cudaMemcpy(cuda_A, A, sizeof(double)*MATRIX_ROWS*MATRIX_COLS, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B, sizeof(double)*MATRIX_ROWS*MATRIX_COLS, cudaMemcpyHostToDevice);

    cal_start = clock();
    MatrixMultOnebyOne<<<grid_size, block_size, 0>>>(cuda_A, cuda_B, cuda_C);
    cudaMemcpy(C, cuda_C, sizeof(double)*MATRIX_ROWS*MATRIX_ROWS, cudaMemcpyDeviceToHost);
    cal_finish = clock();

    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);
    finish = clock();
    double total_err = 0;

    for(int i = 0; i<MATRIX_ROWS*MATRIX_ROWS; i++)
    {
        total_err += fabs(C[i] - C_cpu[i]);
    }
    printf("Mode 3: Total error: %lf, GPU Calculation time: %lf, Total time: %lf \n", total_err, 1.0 * (cal_finish - cal_start) / CLOCKS_PER_SEC, 1.0 * (finish - start) / CLOCKS_PER_SEC);

}
void test_matrix_mult_gpu_shared_memory()
{
    clock_t start, finish, cal_start, cal_finish;
    double *cuda_A, *cuda_B, *cuda_C;

    dim3 block_size(SHARED_BLOCK_SIZE, SHARED_BLOCK_SIZE, 1);
    dim3 grid_size((MATRIX_ROWS+SHARED_BLOCK_SIZE-1)/SHARED_BLOCK_SIZE, (MATRIX_ROWS + SHARED_BLOCK_SIZE -1)/SHARED_BLOCK_SIZE);


    start = clock();
    cudaMalloc((void**)&cuda_A, sizeof(double) * MATRIX_ROWS * MATRIX_COLS);
    cudaMalloc((void**)&cuda_B, sizeof(double) * MATRIX_ROWS * MATRIX_COLS);
    cudaMalloc((void**)&cuda_C, sizeof(double) * MATRIX_ROWS * MATRIX_ROWS);
    cudaMemcpy(cuda_A, A, sizeof(double)*MATRIX_ROWS*MATRIX_COLS, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B, sizeof(double)*MATRIX_ROWS*MATRIX_COLS, cudaMemcpyHostToDevice);

    cal_start = clock();
    MatrixMultSharedMemory<<<grid_size, block_size, 0>>>(cuda_A, cuda_B, cuda_C);
    cudaMemcpy(C, cuda_C, sizeof(double)*MATRIX_ROWS*MATRIX_ROWS, cudaMemcpyDeviceToHost);
    cal_finish = clock();

    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);
    finish = clock();
//    for(int idx = 0; idx < MATRIX_ROWS; idx ++)
//    {
//        for(int jdx = 0; jdx <MATRIX_ROWS; jdx++)
//            printf("%lf %lf ", C[idx * MATRIX_ROWS + jdx], C_cpu[idx * MATRIX_ROWS + jdx]);
//        printf("\n");
//    }

    double total_err = 0;
    for(int i = 0; i<MATRIX_ROWS*MATRIX_ROWS; i++)
    {
        total_err += fabs(C[i] - C_cpu[i]);
    }
    printf("Mode 4: Total error: %lf, GPU Calculation time: %lf, Total time: %lf \n", total_err, 1.0 * (cal_finish - cal_start) / CLOCKS_PER_SEC, 1.0 * (finish - start) / CLOCKS_PER_SEC);


}
void test_matrix_mult_gpu_transpose_B()
{
    clock_t start, finish, cal_start, cal_finish;
    double *cuda_A, *cuda_B, *cuda_C, *cuda_transB;

    start = clock();
    cudaMalloc((void**)&cuda_A, sizeof(double) * MATRIX_ROWS * MATRIX_COLS);
    cudaMalloc((void**)&cuda_B, sizeof(double) * MATRIX_ROWS * MATRIX_COLS);
    cudaMalloc((void**)&cuda_transB, sizeof(double) * MATRIX_ROWS * MATRIX_COLS);
    cudaMalloc((void**)&cuda_C, sizeof(double) * MATRIX_ROWS * MATRIX_ROWS);
    cudaMemcpy(cuda_A, A, sizeof(double)*MATRIX_ROWS*MATRIX_COLS, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B, sizeof(double)*MATRIX_ROWS*MATRIX_COLS, cudaMemcpyHostToDevice);
    TransposeMatrix<<<BLOCK_NUM, THREAD_NUM, 0>>>(cuda_B, cuda_transB);
    cudaMemcpy(transB, cuda_transB, sizeof(double)*MATRIX_ROWS*MATRIX_COLS, cudaMemcpyDeviceToHost);

//    for(int i = 0; i<MATRIX_COLS; i++)
//    {
//        for(int j = 0; j<MATRIX_ROWS; j++)
//            printf("%lf ", B[i * MATRIX_ROWS + j]);
//        printf("\n");
//    }
//
//    for(int i = 0; i<MATRIX_ROWS; i++)
//    {
//        for(int j = 0; j<MATRIX_COLS; j++)
//            printf("%lf ", transB[i * MATRIX_COLS + j]);
//        printf("\n");
//    }

    cal_start = clock();
    MatrixMultTransposeB<<<BLOCK_NUM, THREAD_NUM, 0>>>(cuda_A, cuda_B, cuda_transB, cuda_C);
    cudaMemcpy(C, cuda_C, sizeof(double)*MATRIX_ROWS*MATRIX_ROWS, cudaMemcpyDeviceToHost);
    cal_finish = clock();

    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);
    finish = clock();

    double total_err = 0;
    for(int i = 0; i<MATRIX_ROWS*MATRIX_ROWS; i++)
    {
        total_err += fabs(C[i] - C_cpu[i]);
    }
    printf("Mode 2: Total error: %lf, GPU Calculation time: %lf, Total time: %lf \n", total_err, 1.0 * (cal_finish - cal_start) / CLOCKS_PER_SEC, 1.0 * (finish - start) / CLOCKS_PER_SEC);

}
void test_matrix_mult()
{
    generate_rand_matrix();
//    for(int idx = 0; idx < MATRIX_ROWS; idx ++)
//    {
//        for(int jdx = 0; jdx <MATRIX_COLS; jdx++)
//            printf("%lf ", A[idx * MATRIX_COLS + jdx]);
//        printf("\n");
//    }
//    printf(" \n");
//    for(int idx = 0; idx < MATRIX_COLS; idx ++)
//    {
//        for(int jdx = 0; jdx < MATRIX_ROWS; jdx++)
//            printf("%lf ", B[idx * MATRIX_ROWS + jdx]);
//        printf("\n");
//    }
//    printf(" \n");
    printf("Test Matrix Multiplication(A(%dx%d) * B(%dx%d)): \n", MATRIX_ROWS, MATRIX_COLS, MATRIX_COLS, MATRIX_ROWS);
    test_matrix_mult_cpu();
    test_matrix_mult_gpu_original();
    test_matrix_mult_gpu_transpose_B();
    test_matrix_mult_gpu_one_by_one();
    test_matrix_mult_gpu_shared_memory();
}