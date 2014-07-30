//
//  ChanVeseImplementation.h
//  ChanVeseIos
//
//  Created by Mohammadreza Hosseini on 4/03/2014.
//  Copyright (c) 2014 Mohammadreza Hosseini. All rights reserved.
//
/*
 The process for executing the program;
 
 1-we have to select the upper and lower threshold from points that is selected by the user
 2-Use the matlab program random_walker_example save in the our current folder to generate the probabability of rechaing
 any of the selected points. to make sure this works we have to also introduce some points that are not in our selcted range of upper and lower threshold
 3-export the results to one csv file called pathprobability.csv in the objective c

 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include "PathprobabilityEstimation.h"
class ChanVeseImplementation
{
public:
    cv::Mat phi0;
    cv::Mat sourceImage;
    cv::Mat phi;
    cv::Mat edg;
    cv::Mat GaussianSmoothLabel;
    cv::Mat realedge;
    int blobCounter=0;
    int contour=0;
    double lowerThreshold;
    double upperThreshold;
    cv::Mat blob;
    double **probabilities;
    struct CVsetup
    {
        double dt; // time step
        double h;  // pixel spacing
        double lambda1;
        double lambda2;
        double mu; // contour length weighting parameter
        double nu; // region area weighting parameter
        unsigned int p; // length weight exponent
    };
    struct CVsetup* pCVinputs;
    // Compute gray level averages in foreground and background regions defined by level set function phi
    void GetRegionAverages(cv::Mat img ,
                           double epsilon,
                           double &c1,
                           double &c2);
    
    // Compute coefficients needed in Chan-Vese segmentation algorithm given current level set function
    void GetChanVeseCoefficients(
                                 unsigned int i,
                                 unsigned int j,
                                 double L,
                                 double& F1,
                                 double& F2,
                                 double& F3,
                                 double& F4,
                                 double& F,
                                 double& deltaPhi);
    
    
    // Reinitialize a function to the signed distance function to its zero contour
    /*void ReinitPhi(Image<double>* phiIn,
     Image<double>** psiOut,
     double dt,
     double h,
     unsigned int numIts);*/
    
    // Main segmentation algorithm
    void ChanVeseSegmentation(cv::Mat img);
    double** mainentrance(NSMutableArray *coordinates);
    void ZeroCrossings(unsigned char fg,
                       unsigned char bg);
    void ZeroCrossings2(unsigned char fg,
                       unsigned char bg);
    void blobdetection();
    void neighbours(int,int);
    PathprobabilityEstimation pathEstimation;
    NSMutableArray *coOrdinates;
};


