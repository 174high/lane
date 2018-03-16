#ifndef HISTOGRAM1D_H
#define HISTOGRAM1D_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Histogram1D{
private:
    int histSize[1];
    float hranges[2];
    const float* ranges[1];
    int channels[1];
public:
    Histogram1D(){
        histSize[0] = 256;
        hranges[0] = 0.0;
        hranges[1] = 255.0;
        ranges[0]  = hranges;
        channels[0] = 0;
    }

    cv::MatND getHistogram(const cv::Mat&image, const cv::Mat& mask = cv::Mat())
    {
        cv::MatND  hist;
        cv::calcHist(&image,
                 1,
                 channels,
                 mask,
                 hist,
                 1,
                 histSize,
                 ranges
                 );
        return hist;
    }

    cv::Mat getHistogramImage(const cv::Mat &image, const cv::Mat& mask = cv::Mat())
    {
        cv::MatND hist = getHistogram(image, mask);
        double maxVal = 0;
        double minVal = 0;
        cv::minMaxLoc(hist,&minVal,&maxVal,0,0);
        cv::Mat histImg(histSize[0],histSize[0],CV_8U,cv::Scalar(255));
        int hpt  = static_cast<int>(0.9*histSize[0]);
        for(int h = 0;h<histSize[0];h++){
            float binVal =hist.at<float>(h);
            int intensity = static_cast<int>(binVal*hpt/maxVal);
            cv::line(histImg,cv::Point(h,histSize[0]),
                    cv::Point(h,histSize[0]-intensity),
                    cv::Scalar::all(0));
        }
        return histImg;
    }

    cv::Mat getHistogramImageWithMV(const cv::Mat &image, double mean, double stddev, const cv::Mat& mask = cv::Mat())
    {
        cv::MatND hist = getHistogram(image, mask);
        double maxVal = 0;
        double minVal = 0;
        cv::minMaxLoc(hist,&minVal,&maxVal,0,0);
        cv::Mat histImg(histSize[0],histSize[0],CV_8UC3,cv::Scalar(255,255,255));
        int hpt  = static_cast<int>(0.9*histSize[0]);
        for(int h = 0;h<histSize[0];h++){
            float binVal = hist.at<float>(h);
            int intensity = static_cast<int>(binVal*hpt/maxVal);
            cv::line(histImg,
                     cv::Point(h,histSize[0]),
                     cv::Point(h,histSize[0]-intensity),
                     cv::Scalar::all(0));
        }

        cv::line(histImg,
                 cv::Point(mean,histSize[0]),
                 cv::Point(mean,histSize[0]-hpt),
                 cv::Scalar(0, 0, 255));

        cv::line(histImg,
                 cv::Point(mean-stddev,histSize[0]),
                 cv::Point(mean-stddev,histSize[0]-hpt),
                 cv::Scalar(255, 0, 0));
        cv::line(histImg,
                 cv::Point(mean+stddev,histSize[0]),
                 cv::Point(mean+stddev,histSize[0]-hpt),
                 cv::Scalar(255, 0, 0));
        
        return histImg;
    }
};
#endif // HISTOGRAM1D_H