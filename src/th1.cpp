//???????????,?????????????????
#include <iostream>  
#include <opencv2/opencv.hpp>  
#include <sys/time.h> 
#include <unistd.h>
using namespace cv;

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Wrong args" << std::endl;
        return 1;
    }

    struct  timeval start;
    struct  timeval end;
    double diff;
  
    cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    imshow("testSrc", image);

    if (image.empty())
    {
        std::cout << "read image failure" << std::endl;
        return -1;
    }

    #if 1
    cv::Mat global;
    cv::threshold(image, global, 127, 255, CV_THRESH_BINARY);
    cv::imshow("global-CV_THRESH_BINARY", global);
    //cv::imwrite("global.jpg", global);
    #endif

    cv::threshold(image, global, 127, 255, CV_THRESH_BINARY_INV);
    cv::imshow("global-CV_THRESH_BINARY_INV", global);

    int blockSize = 25;
    int constValue = 10;
    cv::Mat local;
    gettimeofday(&start,NULL);
    cv::adaptiveThreshold(image, local, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);
    gettimeofday(&end,NULL);
    
    diff =  (end.tv_sec-start.tv_sec) + (double)(end.tv_usec-start.tv_usec) / 1000000;
    printf("the difference is %lf sec\n",diff);

    
    //cv::imwrite("local.jpg", local);
    cv::imshow("localThreshold", local);
    cv::waitKey(0);


    return 0;
}
