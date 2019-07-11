//
// Created by admin1 on 19-7-10.
//

#ifndef ARRAYSUM_CUDA_TEST_H
#define ARRAYSUM_CUDA_TEST_H


#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>



#define DATA_SIZE 10000000
#define THREAD_NUM 256
#define BLOCK_NUM 64

void test_array_sum();
void test_matrix_product();


#endif //ARRAYSUM_CUDA_TEST_H
