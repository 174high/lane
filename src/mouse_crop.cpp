#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <stdio.h>  
#include <iostream>  

#include "mouse_crop.h"

//using namespace cv;  

std::string image_name;
size_t crop_count = 0;
//cv::Mat org,dst,img,tmp;

//std::string GenFilename() {
//  return std::string("crop_images/") + GetAbsoluteFilePrefix(image_name) + "_" + std::to_string(crop_count++) + ".jpg";
//}

int main(int argc, char** argv)  
{  
  if (argc != 2) {
    std::cout << "Wrong args" << std::endl;
    return 1;
  }

  //system("mkdir -p crop_images");

  image_name = argv[1];
  cv::Mat org = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);  
  if (org.empty()) {
    std::cout << "read image failure" << std::endl;
    exit(-1);
  }

  MouseCrop mc;
  mc.Run(org, "img");
  
  printf("1111111111111111111\n");

  return 0;
} 
