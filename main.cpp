//
// Created by admin1 on 19-7-10.
//
#include "cuda_test.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
//打印设备信息
void printDeviceProp(const cudaDeviceProp &prop)
{


    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %zd\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %zd.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %zd.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %zd.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %zd.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

//CUDA 初始化
bool InitCUDA()
{
    int count;

    //取得支持Cuda的装置的数目
    cudaGetDeviceCount(&count);

    if (count == 0)
    {
        fprintf(stderr, "There is no device.\n");

        return false;
    }

    int i;

    for (i = 0; i < count; i++)
    {

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        //打印设备信息
        printDeviceProp(prop);

        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
        {
            if (prop.major >= 1)
            {
                break;
            }
        }
    }

    if (i == count)
    {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    cudaSetDevice(i);

    return true;

}

int main()
{
   if(!InitCUDA())
   {
       return 0;
   }
   printf("\nCuda initialized! \n");

//   test_array_sum();
//
//   test_matrix_mult();


   cv::Mat img = cv::imread("/home/admin1/workspace/gipuma/scripts/data/dtu/capture_pic2-0522/0_1.bmp", cv::IMREAD_GRAYSCALE);
//   cv::imshow("img", img);
   // cv::waitKey(10000);
   uchar * s;
   s = (uchar*)malloc(1024*1280*sizeof(uchar)*3);
   memcpy(s, img.data, 1024*1280*sizeof(uchar));
   memcpy(s+1024*1280*sizeof(uchar), img.data, 1024*1280*sizeof(uchar));
   memcpy(s+1024*1280*sizeof(uchar)*2, img.data, 1024*1280*sizeof(uchar));
   cv::Mat img_color = cv::Mat(1024, 1280, CV_8UC3, s);


   cv::Mat m1 = img(cv::Rect(0,0,15,15));
   cv::Mat m2 = img_color(cv::Rect(0,0,5,5));
   std::cout << m1 << std::endl;
   std::cout << m2 << std::endl;




//   cv::imshow("img_color", img_color);
//   std::cout << img.channels() << img_color.channels()<<std::endl;
   cv::waitKey(10000);
}