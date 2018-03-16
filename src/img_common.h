#ifndef IMG_COMMON_H
#define IMG_COMMON_H

#include <cmath>
#include <functional>
#include <opencv2/opencv.hpp>  

struct SOFeature {
  cv::Mat image;
  double mean;
  double stddev;
  int max_prob;
};

//typedef std::function<void (uchar* p, uchar* p_mask)> PixelFunc;

/** @Method 1: the efficient method
 accept grayscale image and RGB image */
// data type:
/*
        C1      C2        C3        C4        C6
uchar	  uchar	  cv::Vec2b	cv::Vec3b	cv::Vec4b
short	  short	  cv::Vec2s	cv::Vec3s	cv::Vec4s
int		  int		  cv::Vec2i	cv::Vec3i	cv::Vec4i
float	  float	  cv::Vec2f	cv::Vec3f	cv::Vec4f	cv::Vec6f
double	double	cv::Vec2d	cv::Vec3d	cv::Vec4d	cv::Vec6d
*/
template <typename T_image, typename T_mask>
inline void ScanImageEfficiet(cv::Mat& image, cv::Mat& mask, std::function<void (T_image* p, T_mask* p_mask)> pf) {
  assert(image.channels() == mask.channels());
  assert(image.rows == mask.rows);
  assert(image.cols == mask.cols);
  
  // channels of the image
  int iChannels = image.channels();
  // rows(height) of the image
  int iRows = image.rows;
  // cols(width) of the image
  int iCols = image.cols * iChannels;

  #if 0
  // check if the image data is stored continuous
  if (image.isContinuous()) {
    iCols *= iRows;
    iRows = 1;
  }
  #endif

  T_image* p;
  T_mask* p_mask;
  for (int i = 0; i < iRows; i++) {
    // get the pointer to the ith row
    p = image.ptr<T_image>(i);
    p_mask = mask.ptr<T_mask>(i);
    // operates on each pixel
    for (int j = 0; j < iCols; j++) {
      // assigns new value
      //p[j] = table[p[j]];
      if (p_mask[j] != 0)
        pf(&p[j], &p_mask[j]);
    }
  }
}

//std::pair<double, double> GetMeanStddev(cv::Mat& gray, cv::Mat& mask, size_t pixel_num) {
inline std::pair<double, double> GetMeanStddev(cv::Mat& gray, cv::Mat& mask) {
  size_t pixel_num = 0;
  size_t total = 0;
  ScanImageEfficiet<uchar, uchar>(gray, mask,
      [&total, &pixel_num]
      (uchar* p, uchar* p_mask) {
        total += *p;
        pixel_num++;
      });
  
  double mean = (double)total/(double)pixel_num;
  //printf("mean: %lf, pixel_num[%lu]\n", mean, pixel_num);

  pixel_num = 0;
  total = 0;
  ScanImageEfficiet<uchar, uchar>(gray, mask,
      //[&mean, &total]
      [&mean, &total, &pixel_num]
      (uchar* p, uchar* p_mask) {
        total += pow(*p-mean, 2);
        pixel_num++;
      });

  double stddev = sqrt((double)total / (double)(pixel_num - 1));
  //printf("size_t max: %lu\n", std::numeric_limits<size_t>::max());
  //printf("total: %lu, (pixel_num - 1):%lu\n", total, (pixel_num - 1));
  //printf("stddev: %lf, pixel_num[%lu]\n", stddev, pixel_num);
  return std::make_pair(mean, stddev);
}

inline SOFeature SelfOrginization(const cv::Mat& origin_gray, cv::Mat& mask, int display_running_histo) {
  cv::Mat gray = origin_gray.clone();
  SOFeature cf;

  double first_stddev = 0.f;
  int MN = 5;
  //size_t pixel_num = gray.rows*gray.cols;
  //cv::Mat mask(gray.size(), gray.type(), cv::Scalar(255));
  for (int iter = 0; iter < MN; iter++) {
    //auto result = GetMeanStddev(gray, mask, pixel_num);
    auto result = GetMeanStddev(gray, mask);
    cf.mean = result.first;
    cf.stddev = result.second;
    Histogram1D h1;
    cv::MatND hist = h1.getHistogram(gray, mask);
    double maxVal = 0;
    cv::Point maxLoc;
    cv::minMaxLoc(hist,0,&maxVal,0,&maxLoc);
    cf.max_prob = maxLoc.y;
  
    if (iter == 0)
      first_stddev = cf.stddev;

    if (display_running_histo) {
      Histogram1D h;
  	  cv::imshow("Histogram-running", h.getHistogramImageWithMV(gray, cf.mean, cf.stddev));
      //cv::imshow("gray", gray);
      char q = cv::waitKey(1000);
      if (q == 'q') exit(-1);
      else if (q == 'p') cv::waitKey(0);
    }

    if (pow(cf.stddev, 2) < first_stddev)
      break;
    
    //pixel_num = gray.rows*gray.cols;
    ScanImageEfficiet<uchar, uchar>(gray, mask,
        //[&cf, &pixel_num]
        [&cf]
        (uchar* p, uchar* p_mask) {
          if (abs(*p - cf.mean) > cf.stddev) {
            *p = 0;
            //pixel_num--;
            *p_mask = 0;
          }
        });
  }

  cf.image = gray;
  return cf;
}

inline cv::Mat ReverseImage(const cv::Mat& origin_gray) {
  cv::Mat gray = origin_gray.clone();
  cv::Mat mask(gray.size(), gray.type(), cv::Scalar(255));
  
  ScanImageEfficiet<uchar, uchar>(gray, mask,
      []
      (uchar* p, uchar* p_mask) {
        *p = 255 - *p;
      });

  return gray;
}

inline std::string GetAbsoluteFilePrefix(const std::string& path) {
  std::string result;
  
  std::size_t found = path.find(".jpg");
  assert(found!=std::string::npos);

  std::size_t pos = path.find_last_of('/');
  if (pos != std::string::npos) {
    result = path.substr(pos + 1, found);
  } else {
    result = path.substr(0, found);
  }

  return result;
}

#endif // IMG_COMMON_H
