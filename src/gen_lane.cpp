#include <iostream>
#include <opencv2/opencv.hpp>  
#include <stdio.h>  

using namespace cv;
using namespace std;

int main(int argc, char *argv[])  
{
  if (argc != 2) {
    std::cout << "Wrong args" << std::endl;
    return 1;
  }
  
	IplImage* img;  
  IplImage* img0;  
  IplImage* img1;  
  IplImage* img2;  
  IplImage* img3; 
  std::string pic_name="test";
  std::string pic_new="new";


 	cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	imshow("testSrc", image);

	if (image.empty())
	{
    std::cout << "read image failure" << std::endl;
    return -1;
	}

	int blockSize = 25;
	int constValue = 10;
	cv::Mat local;
	cv::adaptiveThreshold(image, local, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);

	cv::imshow("localThreshold", local);
  IplImage localTmp = local;
  img = cvCloneImage(&localTmp);
  if (NULL == img)  
    return 0;  
 
  img0 = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
  cvCopy(img, img0, NULL);
  img1 = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1); 
  cvCopy(img0, img1, NULL);
  img2 = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
  img3 = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);

  int height = img1->height;  
  int width = img1->width;  
  int step = img1->widthStep;  
  int channels = img1->nChannels; 
  bool on=true;

  cout<<"size : height="<<img1->height<<"  width="<<img1->width<<" widthStep="<<img1->widthStep<<" nchannels="<<img1->nChannels<<std::endl; 

  uchar *data = (uchar*)img1->imageData;  
  uchar *data2 = (uchar*)img2->imageData; 

  for(int i=0;i != height; ++ i)  
  {  
    for(int j=0;j != width; ++ j)  
    {  
      for(int k=0;k != channels; ++ k)  
      {  
        if(data[i*step+j*channels+k]<128)  
        {
          data[i*step+j*channels+k]=0;//255-data[i*step+j*channels+k];  
        }
        else  
        {
#ifdef test_simple_point1
          data[i*step+j*channels+k]=0;
#else
          data[i*step+j*channels+k]=255;//255-data[i*step+j*channels+k];  
#endif
        }
      }  
    }  
  } 

  cvCopy(img1, img2, NULL);
  cvCopy(img1, img3, NULL);

  int height_start  = 0;
  int height_end    = img1->height ; 
  int width_start   = 0;
  int width_end     = img1->width;

  double xl=0,yl=0 ;

  int num=0; 

  float a=0,b=0;          

	float  dl=0,d=0;

  float new_d=0;

  for(int itera=0;itera<5;itera++)        
  {
    num=0;
    xl=0;
    yl=0;                 
    for(int x=height_start;x !=height_end; ++x)
    {
      for(int y=width_start;y !=width_end; ++y)  
      {
        //              printf("--- test x=%d,y=%d,data=%d \n",x,y,data2[x*step+y]);
        if(255==data[x*step+y])    
        {
          num++;
          xl+=x;
          yl+=y;
          printf(" --- line-point x=%d y=%d xl=%g yl=%g num=%d \n",x,y,xl,yl,num);  
        }
      }
    }

    if(0==num)
      break; 

    double xi=0 , yi=0 ;  

    xi=xl/num ;
    yi=yl/num ; 

    printf("--- c-point xi=%g,yi=%g num=%d \n",xi,yi,num); 


    double ax=0,bx=0;

    for(int x=height_start;x !=height_end; ++x)
    {
      for(int y=width_start;y !=width_end; ++y)
      {
        if(255==data[x*step+y])
        {
          ax+=(x-xi)*x ;
          bx+=(y-yi)*x ;
          printf(" --- line-b x=%d y=%d aa=%f bb=%f  \n",x,y,(x-xi)*x,(y-yi)*x);
          printf(" --- line-b x=%d y=%d ax=%f bx=%f  \n",x,y,ax,bx);
        }
      }
    }


    b=bx/ax;                 

    a=yi-b*xi;

    printf(" ---- linx y= %f+%fx \n", a,b); 

    dl=0; 

    unsigned int dl_num=0;
    float dx=0,dy=0,di;

    for(int x=height_start;x !=height_end; ++x)
    {
      for(int y=width_start;y !=width_end; ++y)
      {
        if(255==data[x*step+y])
        {
          dl_num++;
          dx=abs(b*x-y+a);
          dy=sqrt(b*b+1);
          dl+=dx/dy ;
          //                       printf("--- x=%d,y=%d  distance=%g \n",x,y,dx/dy);
        }
      }
    }

    di=dl/dl_num;

    printf("--- di=%g dl=%g dl_num=%d \n",di,dl,dl_num);


    unsigned int test_x=20,test_y=23; 
    float test_distance=0;


    //      dx=abs(b*test_x-test_y+a);
    dx=b*test_x-test_y+a; 
    dy=sqrt(b*b+1);
    test_distance=dx/dy ;
    // data[15*step+78]=255;

    printf("---- test distance\n");
    printf("---- line y= %f+%fx \n", a,b);
    printf("---- point x=%d,y=%d test_d=%g \n",test_x,test_y,test_distance);


    float si=0,si2=0;
    unsigned s_num=0;

    for(int x=height_start;x !=height_end; ++x)
    {
      for(int y=width_start;y !=width_end; ++y)
      {
        if(255==data[x*step+y])
        {
          dx=b*x-y+a;
          dy=sqrt(b*b+1);
          d=dx/dy ;
          //                       printf("--test-- d=%g \n",d);
          si+=(d-di)*(d-di);
          //                       si+=abs(d-di);
        }
      }
    }

    si2=si/(num-1);

    printf("--- si=%g  si2=%g  \n",si,si2);

    new_d=sqrt(si2);

    //       new_d=si2;

    printf("--- new_d %g \n",new_d); 


    cvCopy(img3, img2, NULL);       


    // data2[15*step+78]=255; 
    // Draw a white line
    for(int x=height_start;x !=height_end; x++)
    {
      for(int y=width_start;y !=width_end; y++)
      {
        if(int(a+b*x)==y)
        {
#ifdef  test_simple_point1
          data2[x*step+y]=255;
#else 
          data2[x*step+y]=255;
#endif 
        }
      }
    }

    cvCopy(img3, img1, NULL);


    // double dx=0,dy=0;
    // Set pixel to 0 beyond the stddev
    for(int x=height_start;x !=height_end; x++)
    {
      for(int y=width_start;y !=width_end; y++)
      {     
        dx=abs(b*x-y+a);
        dy=sqrt(b*b+1);
        d=dx/dy ;      
        if(d>new_d)
          data[x*step+y]=0;
      }
    }

    pic_name+="1" ;
    pic_new+="1";
    cvNamedWindow(pic_name.c_str(), 0 );
    cvShowImage(pic_name.c_str(), img2);
    cvNamedWindow(pic_new.c_str(), 0 );
    cvShowImage(pic_new.c_str(), img1);

  }
  cvWaitKey(0);  

  cvDestroyWindow( "test1" );  

  cvReleaseImage( &img0 );  
  cvReleaseImage( &img1 );  

  return 0;
}  
