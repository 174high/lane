#include <iostream>  
#include <opencv2/opencv.hpp>  
#include <sys/time.h> 
#include <unistd.h>
#include <stdio.h>

using namespace std;
using namespace cv;

#if 0
inline float gamma(float x)
{return x>0.04045?pow((x+0.055f)/1.055f,2.4f):x/12.92;};

void RGBToLab(unsigned char*rgbImg,float*labImg)
{
    float B=gamma(rgbImg[0]/255.0f);
    float G=gamma(rgbImg[1]/255.0f);
    float R=gamma(rgbImg[2]/255.0f);
    float X=0.412453*R+0.357580*G+0.180423*B;
    float Y=0.212671*R+0.715160*G+0.072169*B;
    float Z=0.019334*R+0.119193*G+0.950227*B;

   X /=0.95047;
   Y /=1.0;
   Z /=1.08883;

    float FX = X > 0.008856f ? pow(X,1.0f/3.0f) : (7.787f * X +0.137931f);
    float FY = Y > 0.008856f ? pow(Y,1.0f/3.0f) : (7.787f * Y +0.137931f);
    float FZ = Z > 0.008856f ? pow(Z,1.0f/3.0f) : (7.787f * Z +0.137931f);
    labImg[0] = Y > 0.008856f ? (116.0f * FY - 16.0f) : (903.3f * Y);
    labImg[1] = 500.f * (FX - FY);
    labImg[2] = 200.f * (FY - FZ);
}
#endif

std::string GetPrefixName(const std::string& image_name) {
  std::size_t found = image_name.find(".jpg");
  assert(found!=std::string::npos);

  return image_name.substr(0, found);
}

#if 1
int ConvertOneImage(const std::string& image_name)
{
    struct  timeval start;
    struct  timeval end;
    double diff;
  
    // Load image
    cv::Mat srcImage = cv::imread(image_name);
    cv::Mat dstImage;
    if (srcImage.empty()) {
        std::cout << "read image failure" << std::endl;
        return -1;
    }

    //imshow("srcImage", srcImage);
    gettimeofday(&start,NULL);
    cv::cvtColor(srcImage, dstImage, cv::COLOR_BGR2Lab);
    gettimeofday(&end,NULL);
    
    diff =  (end.tv_sec-start.tv_sec) + (double)(end.tv_usec-start.tv_usec) / 1000000;
    //printf("the difference is %.3lf sec\n",diff);
    char diff_buff[16];
    sprintf(diff_buff, "%.3lfs", diff);

    vector<Mat> channels;
    split(dstImage,channels);
    Mat L = channels.at(0);
    Mat A = channels.at(1);
    Mat B = channels.at(2);

    std::string dstImage_name_L = std::string("lab_images/") + "l_" + GetPrefixName(image_name) + "_" + std::string(diff_buff) + ".jpg";
    std::string dstImage_name_A = std::string("lab_images/") + "a_" + GetPrefixName(image_name) + "_" + std::string(diff_buff) + ".jpg";
    std::string dstImage_name_B = std::string("lab_images/") + "b_" + GetPrefixName(image_name) + "_" + std::string(diff_buff) + ".jpg";
    cv::imwrite(dstImage_name_L, L);
    cv::imwrite(dstImage_name_A, A);
    cv::imwrite(dstImage_name_B, B);
    //cv::waitKey(0);
    cv::imwrite(std::string("lab_images/") + image_name, srcImage);

    return 0;
}

int main(int argc, char** argv)
{
    //if (argc != 2) {
    //    std::cout << "Wrong args" << std::endl;
    //    return 1;
    //}

  system("rm -rf lab_images");
  system("mkdir lab_images");

  std::vector<std::string> images = {"0.jpg", "1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg" };
  for (const auto& m: images) {
    ConvertOneImage(m);
  }
  return 0;
}
#else
int main( int argc, char** argv ) 
{ 
    VideoCapture cap("test.mp4"); 
    if(!cap.isOpened()) {
        printf("Open video file failed\n");
        return -1; 
    }
    
    Mat frame;  
    while(true) { 
        cap>>frame; 
        if(frame.empty()) break;
        imshow("video", frame); 
        if(waitKey(30) >=0) 
            break;
    } 
    return 0;
}
#endif
