//
//  PathProbabilityCalculation.cpp
//  ChanVeseIos
//
//  Created by Mohammadreza Hosseini on 27/05/2014.
//  Copyright (c) 2014 Mohammadreza Hosseini. All rights reserved.
//

#include "PathprobabilityEstimation.h"
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#include <algorithm>
#include <vector>
SpMat laplacianMatrix(512*512,512*512);



void PathprobabilityEstimation::set(cv::Mat I,double lower, double upper)
{
    img=I;
    upperthreshold=upper;
    lowerthreshold=lower;
}

void PathprobabilityEstimation::createseeds()
{
    /*int X=img.cols;
    int Y=img.rows;
    NSMutableIndexSet *index=[NSMutableIndexSet new];
    for (int i=0; i<Y; i++)
        for(int j=0; j< X; j++)
            if ((int)img.at<uchar>(i,j) <= upperthreshold && (int)img.at<uchar>(i,j) >= lowerthreshold)
            {
                [OutCoordinates addObject:[NSNumber numberWithInt:j*img.rows+i]];
            }
    for (int i=0; i< ClickedCoordinates.count;i++)
        for(int j=0; j< OutCoordinates.count;j++)
        {
            
            int value1=(int)[[ClickedCoordinates objectAtIndex:i] integerValue];
            int value2=(int)[[OutCoordinates objectAtIndex:j] integerValue];
            if (value1==value2)
                 {
                     [index addIndex:j];
                 }
            
        }
    [OutCoordinates removeObjectsAtIndexes:index];
    //[OutCoordinates removeObject:ClickedCoordinates];
    for (int i=0; i< X; i++)
        for(int j=0; j< Y; j++)
            if ((int)img.at<uchar>(i,j) > upperthreshold || (int)img.at<uchar>(i,j) < lowerthreshold)
            {
                [ClickedCoordinates addObject:[NSNumber numberWithInt:j*img.rows+i]];
            }*/
    
    int X=img.cols;
    int Y=img.rows;
    [OutCoordinates removeAllObjects];
    NSMutableIndexSet *index=[NSMutableIndexSet new];
    for (int i=0; i< Y; i++)
        for(int j=0; j< X; j++)
            if ((float)phi.at<float>(i,j) >=0 &&((int)img.at<uchar>(i,j) > upperthreshold || (int)img.at<uchar>(i,j) < lowerthreshold))
            {
                [ClickedCoordinates addObject:[NSNumber numberWithInt:j*img.rows+i]];
            }
    numberofTest=ClickedCoordinates.count;
   for (int i=0; i< Y; i++)
        for(int j=0; j< X; j++)
            if ((float)phi.at<float>(i,j) < 0)
            {
                [ClickedCoordinates addObject:[NSNumber numberWithInt:j*img.rows+i]];
            }
    
    for (int i=0; i<Y; i++)
        for(int j=0; j< X; j++)
            if ((int)img.at<uchar>(i,j) <= upperthreshold && (int)img.at<uchar>(i,j) >= lowerthreshold && (float)phi.at<float>(i,j) >=0)
            {
                [OutCoordinates addObject:[NSNumber numberWithInt:j*img.rows+i]];
            }
    for (int i=0; i< numberofClickedin;i++)
        for(int j=0; j< OutCoordinates.count;j++)
        {
            
            int value1=(int)[[ClickedCoordinates objectAtIndex:i] integerValue];
            int value2=(int)[[OutCoordinates objectAtIndex:j] integerValue];
            if (value1==value2)
            {
                [index addIndex:j];
            }
        }
    [OutCoordinates removeObjectsAtIndexes:index];
  /* for (int i=0; i< Y; i++)
        for(int j=0; j< X; j++)
            if ((float)phi.at<float>(i,j) < 0)
            {
                [ClickedCoordinates addObject:[NSNumber numberWithInt:j*img.rows+i]];
            }
    
    for (int i=0; i<Y; i++)
        for(int j=0; j< X; j++)
            if ((float)phi.at<float>(i,j) >=0)
            {
                [OutCoordinates addObject:[NSNumber numberWithInt:j*img.rows+i]];
            }
    for (int i=0; i< numberofClickedin;i++)
        for(int j=0; j< OutCoordinates.count;j++)
        {
            
            int value1=(int)[[ClickedCoordinates objectAtIndex:i] integerValue];
            int value2=(int)[[OutCoordinates objectAtIndex:j] integerValue];
            if (value1==value2)
            {
                [index addIndex:j];
            }
            
        }
    [OutCoordinates removeObjectsAtIndexes:index];*/
    //NSLog(@"%lu",(OutCoordinates.count)+(ClickedCoordinates.count));
}



double PathprobabilityEstimation::makeweights(NSInteger valuein,NSInteger valueout)
{
    
    double EPSILON = 1e-6;
    int sigma=6;
    int i1=(int)valuein % img.rows;
    int j1=(int)valuein / img.rows;
    
    int i2=(int)valueout % img.rows;
    int j2=(int)valueout / img.rows;
    
    double valDistances=pow(img.at<uchar>(i1,j1)-(lowerthreshold+upperthreshold)/2,2)+pow(img.at<uchar>(i2,j2)-(lowerthreshold+upperthreshold)/2,2);
    //if (phi.at<float>(i1,j1)>=0 && phi.at<float>(i2,j2) >=0)
        return (exp(-(valDistances)/pow(sigma,2))+EPSILON);
    //else
      //  return 0.0000000005;
    
}

void PathprobabilityEstimation::insertCoefficient(int id, int i, int j, std::vector<T>& coeffs,int n)
{
    int id1 = i+j*n;
    if(i!=-1 && i!=n && j!=-1 && j!=n && id1!=id)
    {
        double w=makeweights(id,id1);
        sumoverrow+=w;
        coeffs.push_back(T(id,id1,-w));
    }
    if (id1==id)
    {
        coeffs.push_back(T(id,id1,sumoverrow));
    }
}

void PathprobabilityEstimation::buildProblem(std::vector<T>& coefficients, int n)
{
    for(int j=0; j<n; ++j)
    {
        for(int i=0; i<n; ++i)
        {
            sumoverrow=0;
            int id = i+j*n;
            insertCoefficient(id, i-1,j, coefficients,n);
            insertCoefficient(id, i+1,j, coefficients,n);
            insertCoefficient(id, i,j-1, coefficients,n);
            insertCoefficient(id, i,j+1, coefficients,n);
            insertCoefficient(id, i,j, coefficients,n);
        }
    }
}

void PathprobabilityEstimation::calcLaplacian()

{
    std::vector<T> coefficients;
    buildProblem(coefficients, img.rows);
    laplacianMatrix.setFromTriplets(coefficients.begin(), coefficients.end());
}
double ** PathprobabilityEstimation::Laplacian()
{
    SpMat laplacianMatrixpermutate(img.rows*img.cols,img.rows*img.cols);
    SpMat laplacianMatrixpermutate2(img.rows*img.cols,img.rows*img.cols);
    //laplacianMatrix.setFromTriplets(coefficients.begin(), coefficients.end());
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(img.rows*img.cols);
    perm.setIdentity();
    for(int i=0;i<ClickedCoordinates.count;i++)
        perm.indices().operator()(i)=(int)[[ClickedCoordinates objectAtIndex:i] integerValue];
    for (int j=0;j < OutCoordinates.count; j++)
        perm.indices().operator()(j+ClickedCoordinates.count)=(int)[[OutCoordinates objectAtIndex:j] integerValue];
    laplacianMatrixpermutate=laplacianMatrix.transpose()*perm;
    laplacianMatrixpermutate2=perm.transpose()*laplacianMatrixpermutate;
    //NSLog(@"%f",laplacianMatrix(37114,37626));
    SpMat BT((int)OutCoordinates.count,(int)numberofClickedin);
    SpMat LU((int)OutCoordinates.count,(int)OutCoordinates.count);
    BT=laplacianMatrixpermutate2.block((int)ClickedCoordinates.count, 0,(int) OutCoordinates.count, (int)numberofClickedin);
    //BT=laplacianMatrixpermutate2.bottomLeftCorner((int)OutCoordinates.count, (int)numberofClickedin);
    //int t=0;
    LU=laplacianMatrixpermutate2.block((int)ClickedCoordinates.count, (int)ClickedCoordinates.count, (int)OutCoordinates.count, (int)OutCoordinates.count);
    Eigen::VectorXd b(numberofClickedin);
    for (int i=0; i< numberofClickedin; i++)
    {
        b(i)=1;
    }
    Eigen::VectorXd bb(OutCoordinates.count);
    bb=-BT*b;
    Eigen::SimplicialCholesky<SpMat> chol(LU);  // performs a Cholesky factorization of A
    Eigen::VectorXd x = chol.solve(bb);
    
    double **probabilities=0;//[img.rows][img.cols];
    probabilities = new double *[img.rows];
    for (int h = 0; h < img.rows; h++)
        probabilities[h] = new double[img.cols];
    
   for(int i=0; i< numberofClickedin;i++)
    {
        int i1=(int)[[ClickedCoordinates objectAtIndex:i] integerValue] % img.rows;
        int j1=(int)[[ClickedCoordinates objectAtIndex:i] integerValue] / img.rows;
        probabilities[i1][j1]=1;
    }
    for(int i=(int)numberofClickedin; i< numberofTest;i++)
    {
        int i1=(int)(int)[[ClickedCoordinates objectAtIndex:i] integerValue] % img.rows;
        int j1=(int)(int)[[ClickedCoordinates objectAtIndex:i] integerValue] / img.rows;
        probabilities[i1][j1]=0;
    }
    for(int i=(int)numberofTest; i< ClickedCoordinates.count;i++)
    {
        int i1=(int)(int)[[ClickedCoordinates objectAtIndex:i] integerValue] % img.rows;
        int j1=(int)(int)[[ClickedCoordinates objectAtIndex:i] integerValue] / img.rows;
        probabilities[i1][j1]=0;
    }
    for(int i=0; i< OutCoordinates.count;i++)
    {
        int i1=(int)(int)[[OutCoordinates objectAtIndex:i] integerValue] % img.rows;
        int j1=(int)(int)[[OutCoordinates objectAtIndex:i] integerValue] / img.rows;
        probabilities[i1][j1]=x(i);
    }
    return probabilities;
}

void PathprobabilityEstimation::removerepetitivecoordinates(NSMutableArray *coordinates,cv::Mat phiout)
{
    phi=phiout;
    NSArray *ClickedCoordinates2=[[NSSet  setWithArray:coordinates] allObjects];
    [ClickedCoordinates removeAllObjects];
    for (int i=0; i<ClickedCoordinates2.count; i++)
    {
        NSValue *value=[ClickedCoordinates2 objectAtIndex:i];
        if ((float)phi.at<float>([value CGPointValue].y,[value CGPointValue].x)>=0.0)
        {
            [ClickedCoordinates addObject:[NSNumber numberWithInt:[value CGPointValue].x*img.rows+[value CGPointValue].y]];
        }
    }
    numberofClickedin=[ClickedCoordinates count];
    NSLog(@"%ld",(long)numberofClickedin);
    createseeds();
}
