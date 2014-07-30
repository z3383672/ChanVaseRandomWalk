//
//  UIImageCVMatConverter.h
//  SelfOpenGL2
//
//  Created by Mohammadreza Hosseini on 5/12/2013.
//  Copyright (c) 2013 Mohammadreza Hosseini. All rights reserved.
//

#import <Foundation/Foundation.h>


@interface UIImageCVMatConverter : NSObject

+ (UIImage *)UIImageFromCVMat:(cv::Mat)cvMat;
+ (UIImage *)UIImageFromCVMat:(cv::Mat)cvMat withUIImage:(UIImage*)image;
+ (cv::Mat)cvMatFromUIImage:(UIImage *)image;
+ (cv::Mat)cvMatGrayFromUIImage:(UIImage *)image;
+ (UIImage *)scaleAndRotateImageFrontCamera:(UIImage *)image;
+ (UIImage *)scaleAndRotateImageBackCamera:(UIImage *)image;
+ (IplImage *)CreateIplImageFromUIImage:(UIImage *)image;
@end
