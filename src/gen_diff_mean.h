#ifndef GEN_DIFF_MEAN_H
#define GEN_DIFF_MEAN_H

#include <iostream>  
#include <opencv2/opencv.hpp>  

#include "histo.h"
#include "img_common.h"

struct LaneFeature {
  //int lane_max_prob;
  //int left_diff;
  //int right_diff;
  SOFeature left_nolane;
  SOFeature lane;
  SOFeature right_nolane;
};

// return lane, nolane
inline LaneFeature GenDiffMean(const cv::Mat& image, int display_running_histo) {
  assert(image.channels() == 1);
  
  cv::Mat otsu;
  cv::threshold(image, otsu, 0, 255, CV_THRESH_OTSU);
  //cv::imshow("otsu", otsu);

  LaneFeature f;
  //printf("lane:\n");
  f.lane = SelfOrginization(image, otsu, display_running_histo);

  // Reverse otsu
  cv::Mat otsu_r = ReverseImage(otsu);

  //printf("nolane:\n");
  f.left_nolane = SelfOrginization(image, otsu_r, display_running_histo);
  f.right_nolane = f.left_nolane;

  return f;
}

#endif // GEN_DIFF_MEAN_H
