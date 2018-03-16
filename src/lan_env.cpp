#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>  
#include <sys/time.h> 
#include <unistd.h>
#include <stdio.h>
#include <gflags/gflags.h>

#include "histo.h"
#include "img_common.h"
#include "mouse_crop.h"

struct TwoLanesFeature {
  LaneFeature left_lane;
  LaneFeature right_lane;
};

struct Features {
  cv::Mat BGR;
  
  cv::Mat B;
  cv::Mat Br;
  cv::Mat G;
  cv::Mat Gr;
  cv::Mat R;
  cv::Mat Rr;
  
  SOFeature fB;
  SOFeature fBr;
  SOFeature fG;
  SOFeature fGr;
  SOFeature fR;
  SOFeature fRr;

  TwoLanesFeature lfB;
  TwoLanesFeature lfG;
  TwoLanesFeature lfR;

  double elapsed_time;
};

using namespace std;
using namespace cv;

DEFINE_bool(display_running_histo, false, "display running histo");
DEFINE_bool(display_histo, false, "display histo");
DEFINE_bool(display_channel_feature, false, "display channel feature");
DEFINE_string(weight, "lane.csv", "pretrained weight");
DEFINE_bool(manual, false, "manual specify two lanes");
DEFINE_uint64(learn_num, 60,
    "The number of images to learn.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)(const std::string& video_name);
typedef std::map<std::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const std::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    std::cout << "Available actions:" << std::endl;
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      std::cout << "\t" << it->first << std::endl;
    }
    std::cout << "Unknown action: " << name << std::endl;
    exit(-1);
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

cv::Mat ResizeImage(const cv::Mat& origin_image) {
  // Resize to D1
  cv::Size target_geometry(720, 576);
  cv::Mat image;
  if (origin_image.size() != target_geometry) {
    cv::resize(origin_image, image, target_geometry);
  } else {
    image = origin_image;
  }

  return image;
}

Features ExtractOneImage(const cv::Mat& image, const std::pair<cv::Rect,cv::Rect>& two_lanes) {
  struct  timeval start;
  struct  timeval end;
  gettimeofday(&start,NULL);
  
  Features f;
  f.BGR = image;
  vector<Mat> channels;
  split(image,channels);
  f.B   = channels.at(0);
  f.Br  = ReverseImage(channels.at(0));
  f.G   = channels.at(1);
  f.Gr  = ReverseImage(channels.at(1));
  f.R   = channels.at(2);
  f.Rr  = ReverseImage(channels.at(2));

  {
  //printf("=========================================== B\n");
  cv::Mat mask(f.B.size(), f.B.type(), cv::Scalar(255));
  f.fB  = SelfOrginization(f.B, mask, FLAGS_display_running_histo);
  f.lfB.left_lane = GenDiffMean(f.B(two_lanes.first), FLAGS_display_running_histo);
  f.lfB.right_lane = GenDiffMean(f.B(two_lanes.second), FLAGS_display_running_histo);
  }
  
  {
  //printf("=========================================== Br\n");
  cv::Mat mask(f.B.size(), f.B.type(), cv::Scalar(255));
  f.fBr = SelfOrginization(f.Br, mask, FLAGS_display_running_histo);
  }
  
  {
  //printf("=========================================== G\n");
  cv::Mat mask(f.B.size(), f.B.type(), cv::Scalar(255));
  f.fG  = SelfOrginization(f.G, mask, FLAGS_display_running_histo);
  f.lfG.left_lane = GenDiffMean(f.G(two_lanes.first), FLAGS_display_running_histo);
  f.lfG.right_lane = GenDiffMean(f.G(two_lanes.second), FLAGS_display_running_histo);
  }
  
  {
  //printf("=========================================== Gr\n");
  cv::Mat mask(f.B.size(), f.B.type(), cv::Scalar(255));
  f.fGr = SelfOrginization(f.Gr, mask, FLAGS_display_running_histo);
  }
  
  {
  //printf("=========================================== R\n");
  cv::Mat mask(f.B.size(), f.B.type(), cv::Scalar(255));
  f.fR  = SelfOrginization(f.R, mask, FLAGS_display_running_histo);
  f.lfR.left_lane = GenDiffMean(f.R(two_lanes.first), FLAGS_display_running_histo);
  f.lfR.right_lane = GenDiffMean(f.R(two_lanes.second), FLAGS_display_running_histo);
  }
  
  {
  //printf("=========================================== Rr\n");
  cv::Mat mask(f.B.size(), f.B.type(), cv::Scalar(255));
  f.fRr = SelfOrginization(f.Rr, mask, FLAGS_display_running_histo);
  }

  gettimeofday(&end,NULL);
  f.elapsed_time =  (end.tv_sec-start.tv_sec) + (double)(end.tv_usec-start.tv_usec) / 1000000;
  //printf("the elapsed time is %.3lf sec\n", f.elapsed_time);

  return f;
}

std::string GenFilename(const std::string& type, const std::string& image_name, char* diff_buff) {
  return std::string("feature_images/") + type + "_" + GetAbsoluteFilePrefix(image_name) + "_" + std::string(diff_buff) + ".jpg";
}

void SaveToFile(const std::string& image_name, const Features& f) {
  char diff_buff[16];
  sprintf(diff_buff, "%.3lfs", f.elapsed_time);
#if 1
  cv::imwrite(GenFilename("B", image_name, diff_buff), f.B);
  cv::imwrite(GenFilename("Br", image_name, diff_buff), f.Br);
  cv::imwrite(GenFilename("G", image_name, diff_buff), f.G);
  cv::imwrite(GenFilename("Gr", image_name, diff_buff), f.Gr);
  cv::imwrite(GenFilename("R", image_name, diff_buff), f.R);
  cv::imwrite(GenFilename("Rr", image_name, diff_buff), f.Rr);
#endif
  cv::imwrite(GenFilename("fB", image_name, diff_buff), f.fB.image);
  cv::imwrite(GenFilename("fBr", image_name, diff_buff), f.fBr.image);
  cv::imwrite(GenFilename("fG", image_name, diff_buff), f.fG.image);
  cv::imwrite(GenFilename("fGr", image_name, diff_buff), f.fGr.image);
  cv::imwrite(GenFilename("fR", image_name, diff_buff), f.fR.image);
  cv::imwrite(GenFilename("fRr", image_name, diff_buff), f.fRr.image);
  cv::imwrite(std::string("feature_images/") + image_name, f.BGR);
}

void RunBatch() {
  system("rm -rf feature_images");
  system("mkdir feature_images");

  //cv::namedWindow("Histogram", cv::WINDOW_NORMAL);
  //cv::namedWindow("gray", cv::WINDOW_NORMAL);
  
  //std::vector<std::string> images = { "0.jpg", "1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg" };
  std::vector<std::string> images = { "1.jpg" };
  //std::vector<std::string> images = { "5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg" };
  for (const auto& m: images) {
    // Load image
    cv::Mat origin_image = cv::imread(m);
    if (origin_image.empty()) {
      std::cout << "read image failure" << std::endl;
      exit(-1);
    }

    MouseCrop mc;
    auto two_lanes = mc.Run(origin_image, "origin_image");
  
    Features f = ExtractOneImage(origin_image, two_lanes);
    SaveToFile(m, f);
  }
}

void PrintFeatures(const Features& f) {
  printf(//"[%6.2lf %6.2lf %6.2lf %6.2lf %6.2lf %6.2lf]\t"
         "[%3d %3d %3d %3d %3d %3d]\t"
         "B[%3d %3d %3d %3d %3d %3d]\t"
         "G[%3d %3d %3d %3d %3d %3d]\t"
         "R[%3d %3d %3d %3d %3d %3d]\n",
    f.fB.max_prob, f.fBr.max_prob, f.fG.max_prob, f.fGr.max_prob, f.fR.max_prob, f.fRr.max_prob,
    f.lfB.left_lane.left_nolane.max_prob, f.lfB.left_lane.lane.max_prob, f.lfB.left_lane.right_nolane.max_prob,
    f.lfB.right_lane.left_nolane.max_prob, f.lfB.right_lane.lane.max_prob, f.lfB.right_lane.right_nolane.max_prob,
    f.lfG.left_lane.left_nolane.max_prob, f.lfG.left_lane.lane.max_prob, f.lfG.left_lane.right_nolane.max_prob,
    f.lfG.right_lane.left_nolane.max_prob, f.lfG.right_lane.lane.max_prob, f.lfG.right_lane.right_nolane.max_prob,
    f.lfR.left_lane.left_nolane.max_prob, f.lfR.left_lane.lane.max_prob, f.lfR.left_lane.right_nolane.max_prob,
    f.lfR.right_lane.left_nolane.max_prob, f.lfR.right_lane.lane.max_prob, f.lfR.right_lane.right_nolane.max_prob);
}

void RunOne(const std::string& image_name) {
  // Load image
  cv::Mat origin_image = cv::imread(image_name);
  if (origin_image.empty()) {
    std::cout << "read image failure" << std::endl;
    exit(-1);
  }

  auto resized_image = ResizeImage(origin_image);
  MouseCrop mc;
  auto two_lanes = mc.Run(resized_image, "resized_image");

  if (FLAGS_display_running_histo) {
    cv::namedWindow("Histogram-running", cv::WINDOW_NORMAL);
  }

  Features f = ExtractOneImage(resized_image, two_lanes);
  if (FLAGS_display_running_histo) {
    cv::destroyWindow("Histogram-running");
  }

  if (FLAGS_display_channel_feature) {
    cv::namedWindow("BGR", cv::WINDOW_NORMAL);
    cv::namedWindow("fB", cv::WINDOW_NORMAL);
    cv::namedWindow("fBr", cv::WINDOW_NORMAL);
    cv::namedWindow("fG", cv::WINDOW_NORMAL);
    cv::namedWindow("fGr", cv::WINDOW_NORMAL);
    cv::namedWindow("fR", cv::WINDOW_NORMAL);
    cv::namedWindow("fRr", cv::WINDOW_NORMAL);

    cv::imshow("BGR", f.BGR);
    cv::imshow("fB", f.fB.image);
    cv::imshow("fBr", f.fBr.image);
    cv::imshow("fG", f.fG.image);
    cv::imshow("fGr", f.fGr.image);
    cv::imshow("fR", f.fR.image);
    cv::imshow("fRr", f.fRr.image);
  }

  if (FLAGS_display_histo) {
    cv::namedWindow("Histogram-fB", cv::WINDOW_NORMAL);
    cv::namedWindow("Histogram-fBr", cv::WINDOW_NORMAL);
    cv::namedWindow("Histogram-fG", cv::WINDOW_NORMAL);
    cv::namedWindow("Histogram-fGr", cv::WINDOW_NORMAL);
    cv::namedWindow("Histogram-fR", cv::WINDOW_NORMAL);
    cv::namedWindow("Histogram-fRr", cv::WINDOW_NORMAL);
    Histogram1D h;
    cv::imshow("Histogram-fB", h.getHistogramImageWithMV(f.fB.image, f.fB.mean, f.fB.stddev));
    cv::imshow("Histogram-fBr", h.getHistogramImageWithMV(f.fBr.image, f.fBr.mean, f.fBr.stddev));
    cv::imshow("Histogram-fG", h.getHistogramImageWithMV(f.fG.image, f.fG.mean, f.fG.stddev));
    cv::imshow("Histogram-fGr", h.getHistogramImageWithMV(f.fGr.image, f.fGr.mean, f.fGr.stddev));
    cv::imshow("Histogram-fR", h.getHistogramImageWithMV(f.fR.image, f.fR.mean, f.fR.stddev));
    cv::imshow("Histogram-fRr", h.getHistogramImageWithMV(f.fRr.image, f.fRr.mean, f.fRr.stddev));
  }

  printf("=========================================== Summary\n");
  PrintFeatures(f);

  cv::waitKey(0);
  cv::destroyAllWindows();
}

#define FEATURE_SIZE  24
std::pair<cv::Rect,cv::Rect> two_lanes;
std::vector<std::pair<std::array<int, FEATURE_SIZE>,std::array<double, FEATURE_SIZE>> > learn_results;

bool read_weight_file(const std::string& fname) {
  bool result = false;
  std::ifstream in(fname);
  std::string line;

  if (in.is_open()) {
    while (std::getline(in, line)) {
      if ((line.length()!=0) && (line.at(0)=='f'))
        continue;
      char* ptr = (char*)line.c_str();
      int len = line.length();
      char* start = ptr;
      int comma_number = 0;
      std::array<int, FEATURE_SIZE> prob_datas_result;
      std::array<double, FEATURE_SIZE> prob_data_stddevs_result;
      for (int i = 0; i < len; i++) {
        if (ptr[i] == ',') {
          if (comma_number < FEATURE_SIZE) {
            prob_datas_result[comma_number] = atoi(start);
          } else {
            prob_data_stddevs_result[comma_number-FEATURE_SIZE] = atof(start);
          }
          start = ptr + i + 1;
          comma_number++;
        }
      }
      if (comma_number != (FEATURE_SIZE*2-1))
        printf("Error: comma_number=%d\n", comma_number);
      assert(comma_number == (FEATURE_SIZE*2-1));
      prob_data_stddevs_result[comma_number-FEATURE_SIZE] = atof(start);
      learn_results.push_back(std::make_pair(prob_datas_result,prob_data_stddevs_result));
    }
    in.close();
    result = true;
  }
  return result;
}

bool write_weight_file(const std::string& fname) {
  std::ofstream ofs;
  ofs.open(fname, std::ofstream::out | std::ofstream::trunc);
  assert(ofs.is_open());
  ofs << "f.fB.max_prob, f.fBr.max_prob, f.fG.max_prob, f.fGr.max_prob, f.fR.max_prob, f.fRr.max_prob,"
    "f.lfB.left_lane.left_nolane.max_prob, f.lfB.left_lane.lane.max_prob, f.lfB.left_lane.right_nolane.max_prob,"
    "f.lfB.right_lane.left_nolane.max_prob, f.lfB.right_lane.lane.max_prob, f.lfB.right_lane.right_nolane.max_prob,"
    "f.lfG.left_lane.left_nolane.max_prob, f.lfG.left_lane.lane.max_prob, f.lfG.left_lane.right_nolane.max_prob,"
    "f.lfG.right_lane.left_nolane.max_prob, f.lfG.right_lane.lane.max_prob, f.lfG.right_lane.right_nolane.max_prob,"
    "f.lfR.left_lane.left_nolane.max_prob, f.lfR.left_lane.lane.max_prob, f.lfR.left_lane.right_nolane.max_prob,"
    "f.lfR.right_lane.left_nolane.max_prob, f.lfR.right_lane.lane.max_prob, f.lfR.right_lane.right_nolane.max_prob,";

  ofs << "f.fB.stddev, f.fBr.stddev, f.fG.stddev, f.fGr.stddev, f.fR.stddev, f.fRr.stddev,"
    "f.lfB.left_lane.left_nolane.stddev, f.lfB.left_lane.lane.stddev, f.lfB.left_lane.right_nolane.stddev,"
    "f.lfB.right_lane.left_nolane.stddev, f.lfB.right_lane.lane.stddev, f.lfB.right_lane.right_nolane.stddev,"
    "f.lfG.left_lane.left_nolane.stddev, f.lfG.left_lane.lane.stddev, f.lfG.left_lane.right_nolane.stddev,"
    "f.lfG.right_lane.left_nolane.stddev, f.lfG.right_lane.lane.stddev, f.lfG.right_lane.right_nolane.stddev,"
    "f.lfR.left_lane.left_nolane.stddev, f.lfR.left_lane.lane.stddev, f.lfR.left_lane.right_nolane.stddev,"
    "f.lfR.right_lane.left_nolane.stddev, f.lfR.right_lane.lane.stddev, f.lfR.right_lane.right_nolane.stddev\n";
  for (const auto& m: learn_results) {
    const auto& prob_data_l = m.first;
    for (size_t i = 0; i < prob_data_l.size(); i++) {
      ofs << prob_data_l[i] << ",";
    }
    const auto& prob_data_stddev_l = m.second;
    for (size_t i = 0; i < prob_data_stddev_l.size(); i++) {
      if (i < (prob_data_stddev_l.size()-1))
        ofs << prob_data_stddev_l[i] << ",";
      else
        ofs << prob_data_stddev_l[i] << "\n";
    }
  }
  return true;
}

int train(const std::string& video_name) {
  read_weight_file(FLAGS_weight);
  cv::VideoCapture cap(video_name);// open the default camera
	if(!cap.isOpened()) {
    printf("Failed to open video\n");
  	exit(-1);
	}

  const cv::Mat prob_data(1,FLAGS_learn_num,CV_8UC1,cv::Scalar(0));
  std::array<cv::Mat, FEATURE_SIZE> prob_datas;
  for (auto& m: prob_datas) {
    m = prob_data.clone();
  }
  const cv::Mat prob_data_stddev(1,FLAGS_learn_num,CV_64FC1,cv::Scalar(0.f));
  std::array<cv::Mat, FEATURE_SIZE> prob_data_stddevs;
  for (auto& m: prob_data_stddevs) {
    m = prob_data_stddev.clone();
  }
  
	cv::namedWindow("video", cv::WINDOW_NORMAL);
  cv::namedWindow("f.fB.image", cv::WINDOW_NORMAL);
  cv::namedWindow("f.fG.image", cv::WINDOW_NORMAL);
  cv::namedWindow("f.fR.image", cv::WINDOW_NORMAL);
  cv::namedWindow("otsu", cv::WINDOW_NORMAL);
  bool first_frame = true;
  size_t index = 0;
	for (index = 0; index < FLAGS_learn_num; ) {
		cv::Mat frame;
		cap >> frame;
    if (frame.empty())
      break;

    auto resized_image = ResizeImage(frame);
    if (first_frame && !FLAGS_manual) {
      MouseCrop mc;
      two_lanes = mc.Run(resized_image, "resized_image");
      first_frame = false;
    } else if (FLAGS_manual) {
      cv::imshow("video", resized_image);
      char k = cv::waitKey();
      if (k == 's') {
        continue;
      } else {
        MouseCrop mc;
        two_lanes = mc.Run(resized_image, "resized_image");
      }
    }

    Features f = ExtractOneImage(resized_image, two_lanes);

    cv::rectangle(resized_image,two_lanes.first,cv::Scalar(0,255,0,0),1,8,0);
    cv::rectangle(resized_image,two_lanes.second,cv::Scalar(0,255,0,0),1,8,0);
    cv::imshow("video", resized_image);
    
		cv::imshow("f.fB.image", f.fB.image);
    cv::moveWindow("f.fB.image", 500, 0);
    cv::imshow("f.fG.image", f.fG.image);
    cv::moveWindow("f.fG.image", 0, 300);
    cv::imshow("f.fR.image", f.fR.image);
    cv::moveWindow("f.fR.image", 500, 300);

    cv::Mat grayImage;
    cv::cvtColor(resized_image, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat otsu;
    cv::threshold(grayImage, otsu, 0, 255, CV_THRESH_OTSU);
    cv::imshow("otsu", otsu);
    cv::moveWindow("otsu", 1100, 0);

    cv::imshow("B(two_lanes.first)", f.B(two_lanes.first));
    cv::moveWindow("B(two_lanes.first)", 900, 0);
    cv::imshow("B(two_lanes.second)", f.B(two_lanes.second));
    cv::moveWindow("B(two_lanes.second)", 900, 150);
    cv::imshow("G(two_lanes.first)", f.G(two_lanes.first));
    cv::moveWindow("G(two_lanes.first)", 400, 300);
    cv::imshow("G(two_lanes.second)", f.G(two_lanes.second));
    cv::moveWindow("G(two_lanes.second)", 400, 450);
    cv::imshow("R(two_lanes.first)", f.R(two_lanes.first));
    cv::moveWindow("R(two_lanes.first)", 900, 300);
    cv::imshow("R(two_lanes.second)", f.R(two_lanes.second));
    cv::moveWindow("R(two_lanes.second)", 900, 450);

    PrintFeatures(f);

    prob_datas[0].at<uchar>(0,index) = f.fB.max_prob;
    prob_datas[1].at<uchar>(0,index) = f.fBr.max_prob;
    prob_datas[2].at<uchar>(0,index) = f.fG.max_prob;
    prob_datas[3].at<uchar>(0,index) = f.fGr.max_prob;
    prob_datas[4].at<uchar>(0,index) = f.fR.max_prob;
    prob_datas[5].at<uchar>(0,index) = f.fRr.max_prob;
    prob_datas[6].at<uchar>(0,index) = f.lfB.left_lane.left_nolane.max_prob;
    prob_datas[7].at<uchar>(0,index) = f.lfB.left_lane.lane.max_prob;
    prob_datas[8].at<uchar>(0,index) = f.lfB.left_lane.right_nolane.max_prob;
    prob_datas[9].at<uchar>(0,index) = f.lfB.right_lane.left_nolane.max_prob;
    prob_datas[10].at<uchar>(0,index) = f.lfB.right_lane.lane.max_prob;
    prob_datas[11].at<uchar>(0,index) = f.lfB.right_lane.right_nolane.max_prob;

    prob_datas[12].at<uchar>(0,index) = f.lfG.left_lane.left_nolane.max_prob;
    prob_datas[13].at<uchar>(0,index) = f.lfG.left_lane.lane.max_prob;
    prob_datas[14].at<uchar>(0,index) = f.lfG.left_lane.right_nolane.max_prob;
    prob_datas[15].at<uchar>(0,index) = f.lfG.right_lane.left_nolane.max_prob;
    prob_datas[16].at<uchar>(0,index) = f.lfG.right_lane.lane.max_prob;
    prob_datas[17].at<uchar>(0,index) = f.lfG.right_lane.right_nolane.max_prob;

    prob_datas[18].at<uchar>(0,index) = f.lfR.left_lane.left_nolane.max_prob;
    prob_datas[19].at<uchar>(0,index) = f.lfR.left_lane.lane.max_prob;
    prob_datas[20].at<uchar>(0,index) = f.lfR.left_lane.right_nolane.max_prob;
    prob_datas[21].at<uchar>(0,index) = f.lfR.right_lane.left_nolane.max_prob;
    prob_datas[22].at<uchar>(0,index) = f.lfR.right_lane.lane.max_prob;
    prob_datas[23].at<uchar>(0,index) = f.lfR.right_lane.right_nolane.max_prob;

    prob_data_stddevs[0].at<double>(0,index) = f.fB.stddev;
    prob_data_stddevs[1].at<double>(0,index) = f.fBr.stddev;
    prob_data_stddevs[2].at<double>(0,index) = f.fG.stddev;
    prob_data_stddevs[3].at<double>(0,index) = f.fGr.stddev;
    prob_data_stddevs[4].at<double>(0,index) = f.fR.stddev;
    prob_data_stddevs[5].at<double>(0,index) = f.fRr.stddev;
    prob_data_stddevs[6].at<double>(0,index) = f.lfB.left_lane.left_nolane.stddev;
    prob_data_stddevs[7].at<double>(0,index) = f.lfB.left_lane.lane.stddev;
    prob_data_stddevs[8].at<double>(0,index) = f.lfB.left_lane.right_nolane.stddev;
    prob_data_stddevs[9].at<double>(0,index) = f.lfB.right_lane.left_nolane.stddev;
    prob_data_stddevs[10].at<double>(0,index) = f.lfB.right_lane.lane.stddev;
    prob_data_stddevs[11].at<double>(0,index) = f.lfB.right_lane.right_nolane.stddev;

    prob_data_stddevs[12].at<double>(0,index) = f.lfG.left_lane.left_nolane.stddev;
    prob_data_stddevs[13].at<double>(0,index) = f.lfG.left_lane.lane.stddev;
    prob_data_stddevs[14].at<double>(0,index) = f.lfG.left_lane.right_nolane.stddev;
    prob_data_stddevs[15].at<double>(0,index) = f.lfG.right_lane.left_nolane.stddev;
    prob_data_stddevs[16].at<double>(0,index) = f.lfG.right_lane.lane.stddev;
    prob_data_stddevs[17].at<double>(0,index) = f.lfG.right_lane.right_nolane.stddev;

    prob_data_stddevs[18].at<double>(0,index) = f.lfR.left_lane.left_nolane.stddev;
    prob_data_stddevs[19].at<double>(0,index) = f.lfR.left_lane.lane.stddev;
    prob_data_stddevs[20].at<double>(0,index) = f.lfR.left_lane.right_nolane.stddev;
    prob_data_stddevs[21].at<double>(0,index) = f.lfR.right_lane.left_nolane.stddev;
    prob_data_stddevs[22].at<double>(0,index) = f.lfR.right_lane.lane.stddev;
    prob_data_stddevs[23].at<double>(0,index) = f.lfR.right_lane.right_nolane.stddev;

    index++;
		if (cv::waitKey(30) >= 0)
			break;
	}

  if (index == FLAGS_learn_num)
    printf("Collect enough prob data, learn_num: %lu\n", FLAGS_learn_num);

  std::array<int, FEATURE_SIZE> prob_datas_result;
  std::array<double, FEATURE_SIZE> prob_data_stddevs_result;
  for (size_t i = 0; i < prob_datas.size(); i++) {
    cv::Mat prob_data_mask(prob_data.size(), prob_data.type(), cv::Scalar(255));
    auto cf = SelfOrginization(prob_datas[i], prob_data_mask, false);
    prob_datas_result[i] = cf.max_prob;

    bool pds_found = false;
    double pds = 0.f;
    ScanImageEfficiet<uchar, double>(prob_datas[i], prob_data_stddevs[i],
        [&cf, &pds_found, &pds]
        (uchar* p, double* p_mask) {
          if (*p == cf.max_prob) {
            pds_found = true;
            pds = *p_mask;
          }
        });

    assert(pds_found == true);
    prob_data_stddevs_result[i] = pds;
  }
  //for (size_t i = 0; i < prob_data_stddevs.size(); i++) {
  //  cv::Mat prob_data_mask(prob_data.size(), prob_data.type(), cv::Scalar(255));
  //  auto cf = SelfOrginization(prob_data_stddevs[i], prob_data_mask, false);
  //  prob_data_stddevs_result[i] = cf.max_prob;
  //}

  printf("Final result:\n");
  {
  auto& pdr = prob_datas_result;
  printf("[%3d %3d %3d %3d %3d %3d]\t"
         "B[%3d %3d %3d %3d %3d %3d]\t"
         "G[%3d %3d %3d %3d %3d %3d]\t"
         "R[%3d %3d %3d %3d %3d %3d]\n",
         pdr[0],pdr[1],pdr[2],pdr[3],pdr[4],pdr[5],
         pdr[6],pdr[7],pdr[8],pdr[9],pdr[10],pdr[11],
         pdr[12],pdr[13],pdr[14],pdr[15],pdr[16],pdr[17],
         pdr[18],pdr[19],pdr[20],pdr[21],pdr[22],pdr[23]);
  }
  {
  auto& pdr = prob_data_stddevs_result;
  printf("[%4.1f %4.1f %4.1f %4.1f %4.1f %4.1f]\t"
         "B[%4.1f %4.1f %4.1f %4.1f %4.1f %4.1f]\t"
         "G[%4.1f %4.1f %4.1f %4.1f %4.1f %4.1f]\t"
         "R[%4.1f %4.1f %4.1f %4.1f %4.1f %4.1f]\n",
         pdr[0],pdr[1],pdr[2],pdr[3],pdr[4],pdr[5],
         pdr[6],pdr[7],pdr[8],pdr[9],pdr[10],pdr[11],
         pdr[12],pdr[13],pdr[14],pdr[15],pdr[16],pdr[17],
         pdr[18],pdr[19],pdr[20],pdr[21],pdr[22],pdr[23]);
  }

  learn_results.push_back(std::make_pair(prob_datas_result,prob_data_stddevs_result));
  printf("learn_results contain %lu items\n", learn_results.size());
  write_weight_file(FLAGS_weight);
  cv::destroyAllWindows();
  return 0;
}
RegisterBrewFunction(train);

double GetProbDistance(const std::array<int, FEATURE_SIZE>& prob_data,
                       const std::pair<std::array<int, FEATURE_SIZE>,std::array<double, FEATURE_SIZE>>& p) {
  const auto& prob_data_l = p.first;
  const auto& prob_data_stddev_l = p.second;
  double s_l_total = 0.f;
  for (size_t i = 0; i < prob_data_l.size(); i++) {
    const auto& delta = prob_data_stddev_l[i];
    double sl = (double)abs(prob_data[i] - prob_data_l[i]);
    double s_l = (sl <= delta ? 0.f : (sl-delta));
    s_l_total += pow(s_l, 2);
  }

  return sqrt(s_l_total);
}

#define WithinStddev(i) \
  ((*p <= (prob_data[i]+(int)prob_data_stddev[i])) && (*p >= (prob_data[i]-(int)prob_data_stddev[i])))

cv::Mat ExtractLane(const cv::Mat& image, size_t min_i) {
  std::array<int, FEATURE_SIZE> prob_data = learn_results[min_i].first;
  std::array<double, FEATURE_SIZE> prob_data_stddev = learn_results[min_i].second;
  vector<Mat> channels;
  split(image,channels);

  cv::Mat mask(channels[0].size(), channels[0].type(), cv::Scalar(255));
  ScanImageEfficiet<uchar, uchar>(channels[0], mask,
      [&prob_data,&prob_data_stddev]
      (uchar* p, uchar* p_mask) {
        if (WithinStddev(7) || WithinStddev(10))
          *p_mask = 0;
      });

  ScanImageEfficiet<uchar, uchar>(channels[1], mask,
      [&prob_data,&prob_data_stddev]
      (uchar* p, uchar* p_mask) {
        if (WithinStddev(13) || WithinStddev(16))
          *p_mask = 0;
      });

  ScanImageEfficiet<uchar, uchar>(channels[2], mask,
      [&prob_data,&prob_data_stddev]
      (uchar* p, uchar* p_mask) {
        if (WithinStddev(19) || WithinStddev(22))
          *p_mask = 0;
      });

  /////////////////////////////////////////////////
  ScanImageEfficiet<uchar, uchar>(channels[0], mask,
      []
      (uchar* p, uchar* p_mask) {
        *p = 0;
      });
  ScanImageEfficiet<uchar, uchar>(channels[1], mask,
      []
      (uchar* p, uchar* p_mask) {
        *p = 0;
      });
  ScanImageEfficiet<uchar, uchar>(channels[2], mask,
      []
      (uchar* p, uchar* p_mask) {
        *p = 0;
      });

  Mat lane_image;
  merge(channels,lane_image);
  return lane_image;
}

int test(const std::string& video_name) {
  assert(read_weight_file(FLAGS_weight) == true);
  cv::VideoCapture cap(video_name);// open the default camera
	if(!cap.isOpened()) {
    printf("Failed to open video\n");
  	exit(-1);
	}

  cv::namedWindow("video", cv::WINDOW_NORMAL);
  cv::namedWindow("lane", cv::WINDOW_NORMAL);
  cv::namedWindow("otsu", cv::WINDOW_NORMAL);
  bool first_frame = true;
  while (true) {
    cv::Mat frame;
		cap >> frame;
    if (frame.empty())
      break;

    auto resized_image = ResizeImage(frame);
    if (first_frame) {
      MouseCrop mc;
      two_lanes = mc.Run(resized_image, "resized_image");
      first_frame = false;
    }
    Features f = ExtractOneImage(resized_image, two_lanes);

    std::array<int, FEATURE_SIZE> prob_data;

    prob_data[0] = f.fB.max_prob;
    prob_data[1] = f.fBr.max_prob;
    prob_data[2] = f.fG.max_prob;
    prob_data[3] = f.fGr.max_prob;
    prob_data[4] = f.fR.max_prob;
    prob_data[5] = f.fRr.max_prob;
    prob_data[6] = f.lfB.left_lane.left_nolane.max_prob;
    prob_data[7] = f.lfB.left_lane.lane.max_prob;
    prob_data[8] = f.lfB.left_lane.right_nolane.max_prob;
    prob_data[9] = f.lfB.right_lane.left_nolane.max_prob;
    prob_data[10] = f.lfB.right_lane.lane.max_prob;
    prob_data[11] = f.lfB.right_lane.right_nolane.max_prob;

    prob_data[12] = f.lfG.left_lane.left_nolane.max_prob;
    prob_data[13] = f.lfG.left_lane.lane.max_prob;
    prob_data[14] = f.lfG.left_lane.right_nolane.max_prob;
    prob_data[15] = f.lfG.right_lane.left_nolane.max_prob;
    prob_data[16] = f.lfG.right_lane.lane.max_prob;
    prob_data[17] = f.lfG.right_lane.right_nolane.max_prob;

    prob_data[18] = f.lfR.left_lane.left_nolane.max_prob;
    prob_data[19] = f.lfR.left_lane.lane.max_prob;
    prob_data[20] = f.lfR.left_lane.right_nolane.max_prob;
    prob_data[21] = f.lfR.right_lane.left_nolane.max_prob;
    prob_data[22] = f.lfR.right_lane.lane.max_prob;
    prob_data[23] = f.lfR.right_lane.right_nolane.max_prob;

    size_t min_i = std::numeric_limits<size_t>::max();
    double min_d = std::numeric_limits<double>::max();
    for (size_t i = 0; i < learn_results.size(); i++) {
      auto d = GetProbDistance(prob_data, learn_results[i]);
      printf("%5.1f\t", d);
      if (d < min_d) {
        min_d = d;
        min_i = i;
      }
    }
    printf("[%lu]\n", min_i);

    auto lane_image = ExtractLane(resized_image, min_i);

    cv::rectangle(resized_image,two_lanes.first,cv::Scalar(0,255,0,0),1,8,0);
    cv::rectangle(resized_image,two_lanes.second,cv::Scalar(0,255,0,0),1,8,0);
    cv::imshow("video", resized_image);
    cv::imshow("lane", lane_image);

    cv::Mat grayImage;
    cv::cvtColor(resized_image, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat otsu;
    cv::threshold(grayImage, otsu, 0, 255, CV_THRESH_OTSU);
    cv::imshow("otsu", otsu);
    cv::moveWindow("otsu", 720, 0);

    if (cv::waitKey(30) >= 0)
			break;
  }

  cv::destroyAllWindows();
  return 0;
}
RegisterBrewFunction(test);

int main(int argc, char** argv) {
  gflags::SetVersionString("1.0.0.0");
  gflags::SetUsageMessage("usage: lan_env <train|test> video_name");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "lan_env");
    return 1;
  }

  int result = GetBrewFunction(std::string(argv[1]))(argv[2]);

  gflags::ShutDownCommandLineFlags();
  return result;
}
