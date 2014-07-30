//
//  OSViewController.m
//  ChanVeseIos
//
//  Created by Mohammadreza Hosseini on 4/03/2014.
//  Copyright (c) 2014 Mohammadreza Hosseini. All rights reserved.
//

#import "OSViewController.h"
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/compat.hpp>
#include  "UIImageCVMatConverter.h"
#include  "ChanVeseImplementation.h"

ChanVeseImplementation chase;
cv::Mat graySource;
cv::Mat graySource2;//=cv::Mat(256,256,CV_8UC1);
cv::Mat imageSource;
cv::Mat grayFastDestination;
cv::Mat imageMatDestination;
cv::Mat im_Bw;
NSMutableArray *coordinates;
int counter=0;
NSTimer *nsTimer;

@interface OSViewController ()


@end

@implementation OSViewController
@synthesize result=_result;

- (void)viewDidLoad
{
    [super viewDidLoad];
    coordinates=[[NSMutableArray alloc] init];
    UIPanGestureRecognizer *recognizer1 = [[UIPanGestureRecognizer alloc]initWithTarget:self action:@selector(handlePan1:)];
    UITapGestureRecognizer *recognizer2 = [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(handleTap1:)];
    [_result addGestureRecognizer:recognizer1];
    [_result addGestureRecognizer:recognizer2];
    //UIImage *destination=[UIImage imageNamed:@"1 (5)n.png"];
   // UIImage *destination=[UIImage imageNamed:@"Image_of_trabecular_bone_of_the_spine_by_Quantitative_computed_tomography.jpg"];
    //UIImage *destination=[UIImage imageNamed:@"knee.jpg"];
    //UIImage *destination=[UIImage imageNamed:@"000004_text.jpg"];
    //UIImage *destination=[UIImage imageNamed:@"9de5bb5818ad99dfdc9323ad93bd9d.jpg"];
    UIImage *destination=[UIImage imageNamed:@"mri-image.jpg"];
    //UIImage *destination=[UIImage imageNamed:@"acl tear.jpg"];
    [[self result] setImage:destination];
    imageSource=[UIImageCVMatConverter cvMatFromUIImage:destination];
    graySource=cv::Mat(imageSource.rows,imageSource.cols,CV_8UC1);
    graySource2=cv::Mat(imageSource.rows,imageSource.cols,CV_8UC1);
    
    

    cv::Mat grayAverage;
    cv::Mat grayTemp;
    cv::cvtColor(imageSource, graySource, CV_BGR2GRAY );
    cv::cvtColor(imageSource, graySource2, CV_BGR2GRAY );
    chase.sourceImage=graySource;
    ///////////
    //chase.upperThreshold=195;
    //chase.lowerThreshold=124;
   // chase.mainentrance(graySource);
   // nsTimer= [NSTimer scheduledTimerWithTimeInterval:1
     //                                target:self
       //                            selector:@selector(updateImage2)
         //                         userInfo:nil
           //                      repeats:YES];
   //[[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(segment) name:@"ChanVeseEnded" object:nil];
    
}

-(void)handlePan1:(UIPanGestureRecognizer *)recognizer
{
    CGPoint touchCoordinates;
    touchCoordinates.x=[recognizer locationInView:_result].x;
    touchCoordinates.x=(int)((touchCoordinates.x/768.0)*graySource.cols);
    touchCoordinates.y=[recognizer locationInView:_result].y;
    touchCoordinates.y=(int)((touchCoordinates.y/1024)*graySource.rows);
    [coordinates addObject:[NSValue valueWithCGPoint:touchCoordinates]];
    for(int i=-2;i<=2;i++)
        for(int j=-2;j<=2;j++)
            graySource2.at<uchar>(touchCoordinates.y-i,touchCoordinates.x-j)=255;
    UIImage* destination=[UIImageCVMatConverter UIImageFromCVMat:graySource2];
    [_result setImage:destination];
}

-(void)handleTap1:(UITapGestureRecognizer *)recognizer
{
   //double **prob;
   // cv::Mat test=cv::Mat(graySource.rows,graySource.cols,CV_8UC1);
  // test=graySource.clone();
    chase.mainentrance(coordinates);
    /*for(int i=0; i<512;i++)
        for (int j=0;j<512;j++)
            if (prob[i][j] >=0.5)
            {
                test.at<uchar>(i, j)=255;
            }
    UIImage* destination=[UIImageCVMatConverter UIImageFromCVMat:test];
    [_result setImage:destination];*/
  nsTimer= [NSTimer scheduledTimerWithTimeInterval:1
                                    target:self
                                    selector:@selector(updateImage2)
                                   userInfo:nil
                                    repeats:YES];

}


- (void) updateImage2
{
    counter++;
    cv::Mat test=cv::Mat(graySource.rows,graySource.cols,CV_8UC1);
    test=graySource.clone();
    chase.ChanVeseSegmentation(graySource);
    for(int i=0;i<chase.phi.rows;i++)
        for(int j=0;j<chase.phi.cols;j++)
        {
      if ((float)chase.edg.at<float>(i,j) == (float)255) //       if ((float)chase.realedge.at<float>(i,j) == (float)255)  // //  
            {
                test.at<uchar>(i,j)=255;
            }
        }
    UIImage* destination=[UIImageCVMatConverter UIImageFromCVMat:test];
    [_result setImage:destination];
    
   /*for(int i=0; i<512;i++)
        for (int j=0;j<512;j++)
            if (chase.probabilities[i][j]> 0.5)
            {
                test.at<uchar>(i, j)=255;
            }
    UIImage* destination=[UIImageCVMatConverter UIImageFromCVMat:test];
    [_result setImage:destination];*/
   if(counter > 1000)
    {
        [nsTimer invalidate];
        nsTimer=Nil;
        [[NSNotificationCenter defaultCenter] postNotificationName:@"ChanVeseEnded" object:self];
    }
    
    
    
}
- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
/*
 -(void) segment
 {
 chase.blobdetection();
 
 float max=0;
 float min=1000.0;
 for (int i=0; i<chase.blob.rows; i++) {
 for (int j=0; j<chase.blob.cols; j++) {
 if (chase.blob.at<float>(i,j) > max) {
 max=chase.blob.at<float>(i,j);
 }
 if (chase.blob.at<float>(i,j) < min) {
 min=chase.blob.at<float>(i,j);
 }
 }
 }
 UIImage* destination=[UIImageCVMatConverter UIImageFromCVMat:chase.blob];
 [_result setImage:destination];
 cv::Mat test=cv::Mat(graySource.rows,graySource.cols,CV_8UC4);
 test=imageSource.clone();
 for (int Counter=1; Counter<=max; Counter++)
 {
 
 int histData[256]={};
 int total=0;
 for (int i=0; i<chase.blob.rows; i++)
 for (int j=0; j<chase.blob.cols; j++)
 if (chase.blob.at<float>(i,j)==(float)Counter)
 {
 total++;
 histData[(int)graySource2.at<uchar>(i,j)] ++;
 }
 float sum = 0;
 for (int t=0 ; t< 256 ; t++) sum += t * histData[t];
 float sumB = 0;
 int wB = 0;
 int wF = 0;
 float varMax = 0;
 int threshold = 0;
 for (int t=0 ; t< 256 ; t++)
 {
 wB += histData[t];               // Weight Background
 if (wB == 0) continue;
 
 wF = total - wB;                 // Weight Foreground
 if (wF == 0) break;
 
 sumB += (float) (t * histData[t]);
 
 float mB = sumB / wB;            // Mean Background
 float mF = (sum - sumB) / wF;    // Mean Foreground
 
 // Calculate Between Class Variance
 float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);
 
 // Check if new maximum found
 if (varBetween > varMax)
 {
 varMax = varBetween;
 threshold = t;
 }
 }
 for (int i=0; i<chase.blob.rows; i++)
 for (int j=0; j<chase.blob.cols; j++)
 if (chase.blob.at<float>(i,j)==(float)Counter)
 {
 if ((int)graySource2.at<uchar>(i,j) < threshold || chase.realedge.at<float>(i,j)==255)
 {
 test.at<cv::Vec4b>(i,j)[0]=255;
 test.at<cv::Vec4b>(i,j)[1]=255;
 test.at<cv::Vec4b>(i,j)[2]=0;
 }
 }
 }
 for (int i=0; i<test.rows; i++)
 for (int j=0; j<test.cols; j++)
 if (test.at<cv::Vec4b>(i,j)[0] !=255)
 {
 test.at<cv::Vec4b>(i,j)[0]=0;
 test.at<cv::Vec4b>(i,j)[1]=0;
 test.at<cv::Vec4b>(i,j)[2]=0;
 }
 destination=[UIImageCVMatConverter UIImageFromCVMat:test];
 [_result setImage:destination];
 }
*/

////Removing zero gardientbackground start here
/*  grayTemp=graySource.clone();
 for (int i=0; i<graySource.rows; i++)
 {
 for (int j=0; j<graySource.cols; j++)
 {
 if  (abs((int) graySource.at<uchar>(i,j)-(int) grayAverage.at<uchar>(i,j)) < 3)
 grayTemp.at<uchar>(i,j)=0;
 }
 }
 filter2D(grayTemp, grayAverage, ddepth , kernel, anchor, delta, cv::BORDER_DEFAULT );
 for (int i=0; i<graySource.rows; i++)
 {
 for (int j=0; j<graySource.cols; j++)
 {
 if  ((int) grayAverage.at<uchar>(i,j)==0)
 graySource.at<uchar>(i,j)=0;
 }
 }
 ////Removing zero gardient background end here
 
 
 // normalizing the image starts here
 //epsilon=1e-1;
 halfsize1=15;ceil(-norminv(epsilon/2,0,sigma1));
 //halfsize2=ceil(-norminv(epsilon/2,0,sigma2));
 int size2=7;//2*halfsize2+1;
 int sigma1,sigma2;
 sigma1=4;
 sigma2=4;
 cv::Mat gaussian0,gaussian1;
 graySource.convertTo(graySource, CV_32FC1);
 GaussianBlur(graySource, gaussian0, cv::Size(size1,size1), sigma1, sigma1, cv::BORDER_DEFAULT);
 
 cv::Mat num;//=cv::Mat(graySource.rows,graySource.cols,CV_32FC1);
 cv::subtract(graySource, gaussian0, num);
 GaussianBlur(num.mul(num), gaussian1, cv::Size(size2,size2), sigma2, sigma2, cv::BORDER_CONSTANT);
 cv::sqrt(gaussian1, gaussian1);
 cv::Mat ln=num/gaussian1;
 float max=-1000;
 float min=1000.0;
 for (int i=0; i<num.rows; i++) {
 for (int j=0; j<num.cols; j++) {
 if (ln.at<float>(i,j) > max) {
 max=ln.at<float>(i,j);
 }
 if (ln.at<float>(i,j) < min) {
 min=ln.at<float>(i,j);
 }
 }
 }
 for (int i=0; i<num.rows; i++) {
 for (int j=0; j<num.cols; j++) {
 ln.at<float>(i,j)=((ln.at<float>(i,j)-min)/(max-min))*255.0;
 }
 }
 ln.convertTo(ln, CV_8UC1);
 graySource= ln.clone();
 for (int i=0; i<graySource.rows; i++)
 {
 for (int j=0; j<graySource.cols; j++)
 {
 if  ((int) grayAverage.at<uchar>(i,j)==0)
 graySource.at<uchar>(i,j)=0;
 }
 }
 
 
 destination=[UIImageCVMatConverter UIImageFromCVMat:graySource];
 // UIImage *destination1=[UIImage imageNamed:@"1 (6)n.png"];
 [[self result] setImage:destination];
 //normalizing the image ends here
 */

/*     //slecting the uuper and lowe threshold authomatcally by machine starts here
 int histData[256]={};
 int total=0;
 for (int i=0; i<graySource.rows; i++)
 for (int j=0; j<graySource.cols; j++)
 {
 if((int)graySource.at<uchar>(i,j) > 0)
 {
 total++;
 histData[(int)graySource.at<uchar>(i,j)] ++;
 }
 }
 float sum = 0;
 for (int t=0 ; t< 256 ; t++) sum += t * histData[t];
 float sumB = 0;
 int wB = 0;
 int wF = 0;
 float varMax = 0;
 for (int t=0 ; t< 256 ; t++)
 {
 wB += histData[t];               // Weight Background
 if (wB == 0) continue;
 
 wF = total - wB;                 // Weight Foreground
 if (wF == 0) break;
 
 sumB += (float) (t * histData[t]);
 
 float mB = sumB / wB;            // Mean Background
 float mF = (sum - sumB) / wF;    // Mean Foreground
 
 // Calculate Between Class Variance
 float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);
 
 // Check if new maximum found
 if (varBetween > varMax)
 {
 varMax = varBetween;
 //threshold = t;
 chase.upperThreshold=220;
 chase.lowerThreshold=1;
 }
 }
 */
//slecting the uuper and lowe threshold authomatcally by machine ends here

/*//Caclutaing the gradient starts here
cv::Mat grad_x, grad_y;
cv::Mat abs_grad_x, abs_grad_y;
int scale2 = 1;
int delta2 = 0;
int ddepth2 = CV_16S;
Sobel( graySource, grad_x, ddepth2, 1, 0, 3, scale2, delta2, cv::BORDER_DEFAULT );
convertScaleAbs( grad_x, abs_grad_x );

/// Gradient Y
//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
Sobel( graySource, grad_y, ddepth2, 0, 1, 3, scale2, delta2, cv::BORDER_DEFAULT );
convertScaleAbs( grad_y, abs_grad_y );


/// Total Gradient (approximate)
cv::Mat grad;
addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
im_Bw=cv::Mat(grad.rows,grad.cols,CV_8UC1);
//cv::GaussianBlur(grad, grad, cv::Size(5,5), 0);
for(int i=0;i<grad.rows;i++)
for(int j=0;j<grad.cols;j++)
{
    if ((int)(grad.at<uchar>(i,j)) < 80)
        im_Bw.at<uchar>(i,j)=0;
        else
            im_Bw.at<uchar>(i,j)=255;
            
            }*/
//cv::Mat GaussianSmoothLabel=cv::Mat(graySource.rows,graySource.cols,CV_32FC1);
//GaussianSmoothLabel.at<float>(271,258)=255;
//cv::GaussianBlur(GaussianSmoothLabel, GaussianSmoothLabel, cv::Size(5,5), 0.5);
// destination=[UIImageCVMatConverter UIImageFromCVMat:GaussianSmoothLabel];
// [_result setImage:destination];

//calculating the gradient ends here
/*////Image avergae intensity calculation begins here
int kernel_size = 3 ;
int ddepth = -1;
cv::Point anchor = cv::Point( -1, -1 );
double delta = 0;
cv::Mat kernel = cv::Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
filter2D(graySource, grayAverage, ddepth , kernel, anchor, delta, cv::BORDER_DEFAULT );
//// Image avergae intensity calculation endss here
*/

