//
//  PathProbabilityCalculation.h
//  ChanVeseIos
//
//  Created by Mohammadreza Hosseini on 27/05/2014.
//  Copyright (c) 2014 Mohammadreza Hosseini. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "Eigen/Sparse"
#include "Eigen/Sparse"

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

class PathprobabilityEstimation
{
public:
    void removerepetitivecoordinates(NSMutableArray *coordinates,cv::Mat);
    void createseeds();
    double makeweights(NSInteger,NSInteger);
    double ** Laplacian();
    NSMutableArray *ClickedCoordinates=[[NSMutableArray alloc] init];
    NSMutableArray *OutCoordinates=[[NSMutableArray alloc] init];
    NSMutableArray *edges=[[NSMutableArray alloc] init];
    NSMutableArray *weights=[[NSMutableArray alloc] init];
    int *boundary;
    void set(cv::Mat, double,double);
    double *LU1D;
    double *BTranspose1D;
    void buildProblem(std::vector<T>& coefficients, int n);
    void insertCoefficient(int id, int i, int j, std::vector<T>& coeffs,int n);
    void calcLaplacian();
    cv::Mat phi;
private:
    cv::Mat img;
    double lowerthreshold;
    double upperthreshold;
    double sumoverrow;
    NSInteger numberofClickedin;
    NSInteger numberofTest;
};



