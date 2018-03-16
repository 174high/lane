#ifndef MOUSE_CROP_H
#define MOUSE_CROP_H

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <stdio.h>  

#include "gen_diff_mean.h"

class MouseCrop {
  private:
    cv::Mat org,dst,img,tmp;
    std::string winname;
    //LaneFeature lf;
    bool left_done;
    bool right_done;
    cv::Rect left_bbox;
    cv::Rect right_bbox;

  public:
    MouseCrop() {}

    std::pair<cv::Rect,cv::Rect> Run(const cv::Mat& image, const std::string& window) {
      left_done = false;
      right_done = false;
      
      org = image.clone();
      org.copyTo(img);  
      org.copyTo(tmp);  
      //winname = window + " (left-lane)";
      winname = window;

      cv::namedWindow(winname);
      cv::setMouseCallback(winname, &MouseCrop::on_mouse, this);
      cv::imshow(winname,img);
      //cv::waitKey(0);
      //cv::destroyWindow(winname);
      //cv::destroyWindow("dst");

      //winname = window + " (right-lane)";
      //cv::namedWindow(winname);
      //cv::setMouseCallback(winname, &MouseCrop::on_mouse, this);
      //cv::imshow(winname,img);  
      cv::waitKey(0);
      cv::destroyWindow(winname);
      cv::destroyWindow("left-lane");
      cv::destroyWindow("right-lane");

      assert(left_done && right_done);
      return std::make_pair(left_bbox, right_bbox);
    }

    static void on_mouse(int event,int x,int y,int flags,void *ustc) {
      MouseCrop* mc = (MouseCrop*)ustc;
      mc->on_mouse(event, x, y, flags);
    }

    void on_mouse(int event,int x,int y,int flags)
    {
        static cv::Point pre_pt = cv::Point(-1,-1);
        static cv::Point cur_pt = cv::Point(-1,-1);
        char temp[16];  
        if (event == CV_EVENT_LBUTTONDOWN)
        {
            org.copyTo(img);
            sprintf(temp,"(%d,%d)",x,y);  
            pre_pt = cv::Point(x,y);  
            putText(img,temp,pre_pt,cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,0,255),1,8);
            circle(img,pre_pt,2,cv::Scalar(255,0,0,0),CV_FILLED,CV_AA,0);
            imshow(winname,img);  
        }  
        else if (event == CV_EVENT_MOUSEMOVE && !(flags & CV_EVENT_FLAG_LBUTTON)) 
        {  
            img.copyTo(tmp);
            sprintf(temp,"(%d,%d)",x,y);  
            cur_pt = cv::Point(x,y);  
            putText(tmp,temp,cur_pt,cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,0,255));
            imshow(winname,tmp);  
        }  
        else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))
        {  
            img.copyTo(tmp);  
            sprintf(temp,"(%d,%d)",x,y);  
            cur_pt = cv::Point(x,y);  
            putText(tmp,temp,cur_pt,cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,0,255));  
            rectangle(tmp,pre_pt,cur_pt,cv::Scalar(0,255,0,0),1,8,0);
            imshow(winname,tmp);  
        }  
        else if (event == CV_EVENT_LBUTTONUP)
        {
            org.copyTo(img);  
            sprintf(temp,"(%d,%d)",x,y);  
            cur_pt = cv::Point(x,y);  
            putText(img,temp,cur_pt,cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,0,255));  
            circle(img,pre_pt,2,cv::Scalar(255,0,0,0),CV_FILLED,CV_AA,0);  
            rectangle(img,pre_pt,cur_pt,cv::Scalar(0,255,0,0),1,8,0);
            imshow(winname,img);  
            img.copyTo(tmp);  
            
            int width = abs(pre_pt.x - cur_pt.x);  
            int height = abs(pre_pt.y - cur_pt.y);  
            if (width == 0 || height == 0)  
            {  
                printf("width == 0 || height == 0");  
                return;  
            }  
            //dst = org(cv::Rect(cv::min(cur_pt.x,pre_pt.x),cv::min(cur_pt.y,pre_pt.y),width,height));
            //cv::namedWindow("dst");
            //cv::imshow("dst",dst);

            if (!left_done) {
              left_bbox = (cv::Rect(cv::min(cur_pt.x,pre_pt.x),cv::min(cur_pt.y,pre_pt.y),width,height));
              dst = org(left_bbox);
              cv::namedWindow("left-lane");
              cv::imshow("left-lane",dst);
              left_done = true;
            } else if (!right_done) {
              right_bbox = (cv::Rect(cv::min(cur_pt.x,pre_pt.x),cv::min(cur_pt.y,pre_pt.y),width,height));
              dst = org(right_bbox);
              cv::namedWindow("right-lane");
              cv::imshow("right-lane",dst);
              right_done = true;
            } else {
              printf("Do nothing\n");
            }
            #if 0
            printf("%s\n", !left_done ? "================== left" : "================== right");
            auto diffmean = GenDiffMean(dst, 0);
            printf("lane[%d], nolane[%d]\n", diffmean.first, diffmean.second);
            if (!left_done) {
              lf.left_lane_max_prob = diffmean.first;
              lf.left_nolane_max_prob = diffmean.second;
              left_done = true;
            } else {
              lf.right_lane_max_prob = diffmean.first;
              lf.right_nolane_max_prob = diffmean.second;
              right_done = true;
            }
            #endif
        }  
    }
};


#endif // MOUSE_CROP_H
