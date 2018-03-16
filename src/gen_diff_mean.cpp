#include <iostream>  
#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>

#include "histo.h"
#include "img_common.h"
#include "gen_diff_mean.h"
#include "mouse_crop.h"

using namespace cv;

DEFINE_bool(display_running_histo, false, "display running histo");

int main(int argc, char** argv) {
  gflags::SetVersionString("1.0.0.0");
  gflags::SetUsageMessage("usage: gen_diff_mean image_name");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 2) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "gen_diff_mean");
    return 1;
  }

  cv::Mat origin_image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  if (origin_image.empty()) {
    std::cout << "read image failure" << std::endl;
    return -1;
  }

  // Get lane bbox
  MouseCrop mc;
  auto two_lanes = mc.Run(origin_image, "origin_image");
  cv::Mat image = origin_image(two_lanes.first);
  cv::imshow("lane-image", image);

  // Display lane histo
  cv::Mat mask_image(image.size(), image.type(), cv::Scalar(255));
  auto result = GetMeanStddev(image, mask_image);
  auto mean = result.first;
  auto stddev = result.second;
  Histogram1D h;
  cv::imshow("lane-Histogram", h.getHistogramImageWithMV(image, mean, stddev));

  if (FLAGS_display_running_histo) {
    cv::namedWindow("Histogram-running", cv::WINDOW_NORMAL);
  }

  LaneFeature lf = GenDiffMean(image, FLAGS_display_running_histo);
  printf("lf.left_nolane.max_prob[%d], lf.lane.max_prob[%d], lf.right_nolane.max_prob[%d]\n",
    lf.left_nolane.max_prob, lf.lane.max_prob, lf.right_nolane.max_prob);

  if (FLAGS_display_running_histo) {
    cv::destroyWindow("Histogram-running");
  }

  cv::waitKey(0);
  gflags::ShutDownCommandLineFlags();
  return 0;
}
