cmake_minimum_required(VERSION 3.0)
set( CMAKE_DEBUG_POSTFIX "d")
add_definitions(-DUNICODE -D_UNICODE)
project(cudaTest)

find_package(CUDA)
if(CUDA_FOUND)
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
else(CUDA_FOUND)
    message(FATAL_ERROR "Cuda Not Found")
endif(CUDA_FOUND)

find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(./)

cuda_add_executable(
        cudaTest
        cuda_test.h
        array_sum.cu
        matrix_mult.cu
        main.cpp
)

