//
//  ChanVeseImplementation.cpp
//  ChanVeseIos
//
//  Created by Mohammadreza Hosseini on 4/03/2014.
//  Copyright (c) 2014 Mohammadreza Hosseini. All rights reserved.
//

#include "ChanVeseImplementation.h"
#include <opencv2/opencv.hpp>
#include "CSVParser.h"


const double PI = 3.14159265358979323846264338327950288;
//const double A=1;
//const double B=83;

//chanvase base method
//hello effect aoundthe biofilm distracted by basic chan -vse
//select based on histogram instead of interaction
//track within farmes


void ChanVeseImplementation::GetRegionAverages(cv::Mat img, double epsilon, double &c1, double &c2)
{
    int height=img.rows;
    int width=img.cols;
    // Non-smoothed calculation
    if (0 == epsilon)
    {
        int n1 = 0;
        int n2 = 0;
        double Iplus = 0;
        double Iminus = 0;
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                if ((float)phi.at<float>(i,j) >= 0)
                {
                    ++n1;
                    Iplus += (int)img.at<uchar>(i,j);// >imageData[i*img->widthStep+j);// data()[i)[j);
                }
                else
                {
                    ++n2;
                    Iminus += (int)img.at<uchar>(i,j);
                }
            }
        }
        c1 = Iplus/double(n1);
        c2 = Iminus/double(n2);
    }
    // Smoothed calculation
    else
    {
        double num1 = 0;
        double den1 = 0;
        double num2 = 0;
        double den2 = 0;
        double H_phi;
        for (int i = 0; i < phi.rows; i++)
        {
            for (int j = 0; j < phi.cols; j++)
            {
                // Compute value of H_eps(phi) where H_eps is a mollified Heavyside function
                
                H_phi = .5*(1+(2/PI)*atan((float)phi.at<float>(i,j)/epsilon));
                num1 += (int)(img.at<uchar>(i,j))*H_phi;
                den1 += H_phi;
                num2 += (int)(img.at<uchar>(i,j))*(1 - H_phi);
                den2 += 1 - H_phi;
            }
        }
        c1 = num1/den1;
        c2 = num2/den2;
        //printf("%f %f",c1,c2);
        //printf("\n");
    }
    
    
}


void ChanVeseImplementation::GetChanVeseCoefficients(
                             
                             unsigned int i,
                             unsigned int j,
                             double L,
                             double& F1,
                             double& F2,
                             double& F3,
                             double& F4,
                             double& F,
                             double& deltaPhi)
{
    // factor to avoid division by zero
    double eps = 0.000001;
    double h = pCVinputs->h;
    double dt = pCVinputs->dt;
    double mu = pCVinputs->mu;
    unsigned int p = pCVinputs->p;
    
    double C1 = 1/sqrt(eps + pow(double(phi.at<float>((i+1),j) - phi.at<float>(i,j)), 2.0)
                       + pow(double(phi.at<float>(i,j+1) - phi.at<float>(i,j-1)),2.0)/4.0);
    double C2 = 1/sqrt(eps + pow(double(phi.at<float>(i,j) - phi.at<float>((i-1),j)),2)
                       + pow(double(phi.at<float>((i-1),j+1) - phi.at<float>((i-1),j-1)),2)/4.0);
    double C3 = 1/sqrt(eps + pow(double(phi.at<float>((i+1),j) - phi.at<float>((i-1),j)),2)/4.0
                       + pow(double(phi.at<float>(i,j+1) - phi.at<float>(i,j)),2));
    double C4 = 1/sqrt(eps + pow(double(phi.at<float>((i+1),j-1) - phi.at<float>((i-1),j-1)),2)/4.0
                       + pow(double(phi.at<float>(i,j) - phi.at<float>(i,j-1)),2));
    
    deltaPhi = h/(PI*(h*h + (phi.at<float>(i,j)*(phi.at<float>(i,j)))));
    
    double Factor = dt*deltaPhi*mu*(double(p)*pow((double)L,(int) (p-1)));
    F = h/(h+Factor*(C1+C2+C3+C4));
    Factor = Factor/(h+Factor*(C1+C2+C3+C4));
    
    F1 = Factor*C1;
    F2 = Factor*C2;
    F3 = Factor*C3;
    F4 = Factor*C4;
}

void ChanVeseImplementation::ZeroCrossings(unsigned char fg,
                   unsigned char bg)
{
    for(int i=0;i< realedge.rows;i++)
        for (int j=0;j<realedge.cols;j++)
            realedge.at<float>(i,j)=(float)bg;
    for (unsigned int i = 0; i < phi.rows; ++i)
    {
        for (unsigned int j = 0; j < phi.cols; ++j)
        {
            // Currently only checking interior pixels to avoid
            // bounds checking
            if (i > 0 && i < (phi.rows-1)
                && j > 0 && j < (phi.cols-1))
            {
                if (0 == phi.at<float>(i,j))
                {
                    if (0 != phi.at<float>(i-1,j-1)
                        || 0 != phi.at<float>(i-1,j)
                        || 0 != phi.at<float>(i-1,j+1)
                        || 0 != phi.at<float>(i,j-1)
                        || 0 != phi.at<float>(i,j+1)
                        || 0 != phi.at<float>(i+1,j-1)
                        || 0 != phi.at<float>(i+1,j)
                        || 0 != phi.at<float>(i+1,j+1))
                    {
                        realedge.at<float>(i,j) = fg;
                    }
                }
                else
                {
                    if (abs(phi.at<float>(i,j)) < abs(phi.at<float>((i-1),j-1))
                        && (phi.at<float>(i,j)>0) != (phi.at<float>((i-1),j-1)>0))
                        realedge.at<float>(i,j) = fg;
                    else if (abs(phi.at<float>(i,j)) < abs(phi.at<float>((i-1),j))
                             && (phi.at<float>(i,j)>0) != (phi.at<float>((i-1),j)>0))
                        realedge.at<float>(i,j) = fg;
                    else if (abs(phi.at<float>(i,j)) < abs(phi.at<float>((i-1),j+1))
                             && (phi.at<float>(i,j)>0) != (phi.at<float>((i-1),j+1)>0))
                        realedge.at<float>(i,j) = fg;
                    else if (abs(phi.at<float>(i,j)) < abs(phi.at<float>(i,j-1))
                             && (phi.at<float>(i,j)>0) != (phi.at<float>(i,j-1)>0))
                        realedge.at<float>(i,j) = fg;
                    else if (abs(phi.at<float>(i,j)) < abs(phi.at<float>(i,j+1))
                             && (phi.at<float>(i,j)>0) != (phi.at<float>(i,j+1)>0))
                        realedge.at<float>(i,j) = fg;
                    else if (abs(phi.at<float>(i,j)) < abs(phi.at<float>((i+1),j-1))
                             && (phi.at<float>(i,j)>0) != (phi.at<float>((i+1),j-1)>0))
                        realedge.at<float>(i,j) = fg;
                    else if (abs(phi.at<float>(i,j)) < abs(phi.at<float>((i+1),j))
                             && (phi.at<float>(i,j)>0) != (phi.at<float>((i+1),j)>0))
                        realedge.at<float>(i,j) = fg;
                    else if (abs(phi.at<float>(i,j)) < abs(phi.at<float>((i+1),j+1))
                             && (phi.at<float>(i,j)>0) != (phi.at<float>((i+1),j+1)>0))
                        realedge.at<float>(i,j) = fg;
                }
            }
        }
    }
}

void ChanVeseImplementation::neighbours(int i, int j)
{
    /*blob.at<float>(i,j)=blobCounter;
    if (edg.at<float>(i,j+1)==255.0 && blob.at<float>(i,j+1)==0.0) {
        neighbours(i, j+1);
    }
    if (edg.at<float>(i+1,j-1)==255.0 && blob.at<float>(i+1,j-1)==0.0) {
        neighbours(i+1, j-1);
    }
    if (edg.at<float>(i+1,j)==255.0 && blob.at<float>(i+1,j)==0.0) {
        neighbours(i+1, j);
    }
    if (edg.at<float>(i+1,j+1)==255.0 && blob.at<float>(i+1,j+1)==0.0) {
        neighbours(i+1, j+1);
    }
    if (edg.at<float>(i,j-1)==255.0 && blob.at<float>(i,j-1)==0.0) {
        neighbours(i, j-1);
    }
    if (edg.at<float>(i-1,j-1)==255.0 && blob.at<float>(i-1,j-1)==0.0) {
        neighbours(i-1, j-1);
    }
    if (edg.at<float>(i-1,j)==255.0 && blob.at<float>(i-1,j)==0.0) {
        neighbours(i-1, j);
    }
    if (edg.at<float>(i-1,j+1)==255.0 && blob.at<float>(i-1,j+1)==0.0) {
        neighbours(i-1, j+1);
    }*/
    
    if (sourceImage.at<uchar>(i,j+1) >=lowerThreshold && sourceImage.at<uchar>(i,j+1) <=upperThreshold && GaussianSmoothLabel.at<float>(i,j+1)==-5)
    {
      
        GaussianSmoothLabel.at<float>(i,j+1)=1;
        neighbours(i,j+1);
    }
    if (sourceImage.at<uchar>(i+1,j-1) >=lowerThreshold && sourceImage.at<uchar>(i+1,j-1) <=upperThreshold && GaussianSmoothLabel.at<float>(i+1,j-1)==-5)
    {
      
        GaussianSmoothLabel.at<float>(i+1,j-1)=1;
        neighbours(i+1, j-1);
    }
    if (sourceImage.at<uchar>(i+1,j) >=lowerThreshold && sourceImage.at<uchar>(i+1,j) <=upperThreshold && GaussianSmoothLabel.at<float>(i+1,j)==-5)
    {
      
        GaussianSmoothLabel.at<float>(i+1,j)=1;
        neighbours(i+1, j);
    }
    if (sourceImage.at<uchar>(i+1,j+1) >=lowerThreshold && sourceImage.at<uchar>(i+1,j+1) <=upperThreshold && GaussianSmoothLabel.at<float>(i+1,j+1)==-5)
    {
       
        GaussianSmoothLabel.at<float>(i+1,j+1)=1;
        neighbours(i+1, j+1);
    }
    if (sourceImage.at<uchar>(i,j-1) >=lowerThreshold && sourceImage.at<uchar>(i,j-1) <=upperThreshold && GaussianSmoothLabel.at<float>(i,j-1)==-5)
    {
       
        GaussianSmoothLabel.at<float>(i,j-1)=1;
        neighbours(i, j-1);
    }
    if (sourceImage.at<uchar>(i-1,j-1) >=lowerThreshold && sourceImage.at<uchar>(i-1,j-1) <=upperThreshold && GaussianSmoothLabel.at<float>(i-1,j-1)==-5)
    {
        
        GaussianSmoothLabel.at<float>(i-1,j-1)=1;
        neighbours(i-1, j-1);
    }
    if (sourceImage.at<uchar>(i-1,j) >=lowerThreshold && sourceImage.at<uchar>(i-1,j) <=upperThreshold && GaussianSmoothLabel.at<float>(i-1,j)==-5)
    {
        GaussianSmoothLabel.at<float>(i-1,j)=1;
        neighbours(i-1, j);
    }
    if (sourceImage.at<uchar>(i-1,j+1) >=lowerThreshold && sourceImage.at<uchar>(i-1,j+1) <=upperThreshold && GaussianSmoothLabel.at<float>(i-1,j+1)==-5)
    {
        GaussianSmoothLabel.at<float>(i-1,j+1)=1;
        neighbours(i-1, j+1);
    }

}

void ChanVeseImplementation::blobdetection()
{
    /*for (int i=0; i<blob.rows; i++)
    {
        for (int j=0; j<blob.cols; j++)
        {
            if (edg.at<float>(i,j)==255.0 && blob.at<float>(i,j)==0.0 )
            {
                blobCounter++;
                neighbours(i,j);
            }
        }
    }*/
    
    
    for (int i=0; i<sourceImage.rows; i++)
    {
        for (int j=0; j<sourceImage.cols; j++)
        {
            if ( GaussianSmoothLabel.at<float>(i,j)== 1 )
            {
                neighbours(i,j);
            }
        }
    }
}

void ChanVeseImplementation::ZeroCrossings2(unsigned char fg,
                                           unsigned char bg)
{
    for(int i=0;i< edg.rows;i++)
        for (int j=0;j<edg.cols;j++)
            edg.at<float>(i,j)=(float)bg;
    
    for (unsigned int i = 1; i < phi.rows-1; ++i)
    {
        for (unsigned int j = 1; j < phi.cols-1; ++j)
        {
            // Currently only checking interior pixels to avoid
            // bounds checking
            if (i > 0 && i < (phi.rows-1)
                && j > 0 && j < (phi.cols-1))
            {
                if (0 == phi.at<float>(i,j))
                {
                    if (0 != phi.at<float>(i-1,j-1)
                        || 0 != phi.at<float>(i-1,j)
                        || 0 != phi.at<float>(i-1,j+1)
                        || 0 != phi.at<float>(i,j-1)
                        || 0 != phi.at<float>(i,j+1)
                        || 0 != phi.at<float>(i+1,j-1)
                        || 0 != phi.at<float>(i+1,j)
                        || 0 != phi.at<float>(i+1,j+1))
                    {
                        edg.at<float>(i,j) = fg;
                    }
                }
                else
                {
                    if (abs(phi.at<float>(i,j)) < abs(phi.at<float>((i-1),j-1))
                        && (phi.at<float>(i,j)>0) != (phi.at<float>((i-1),j-1)>0))
                        edg.at<float>(i,j) = fg;
                    else if (abs(phi.at<float>(i,j)) < abs(phi.at<float>((i-1),j))
                             && (phi.at<float>(i,j)>0) != (phi.at<float>((i-1),j)>0))
                        edg.at<float>(i,j) = fg;
                    else if (abs(phi.at<float>(i,j)) < abs(phi.at<float>((i-1),j+1))
                             && (phi.at<float>(i,j)>0) != (phi.at<float>((i-1),j+1)>0))
                        edg.at<float>(i,j) = fg;
                    else if (abs(phi.at<float>(i,j)) < abs(phi.at<float>(i,j-1))
                             && (phi.at<float>(i,j)>0) != (phi.at<float>(i,j-1)>0))
                        edg.at<float>(i,j) = fg;
                    else if (abs(phi.at<float>(i,j)) < abs(phi.at<float>(i,j+1))
                             && (phi.at<float>(i,j)>0) != (phi.at<float>(i,j+1)>0))
                        edg.at<float>(i,j) = fg;
                    else if (abs(phi.at<float>(i,j)) < abs(phi.at<float>((i+1),j-1))
                             && (phi.at<float>(i,j)>0) != (phi.at<float>((i+1),j-1)>0))
                        edg.at<float>(i,j) = fg;
                    else if (abs(phi.at<float>(i,j)) < abs(phi.at<float>((i+1),j))
                             && (phi.at<float>(i,j)>0) != (phi.at<float>((i+1),j)>0))
                        edg.at<float>(i,j) = fg;
                    else if (abs(phi.at<float>(i,j)) < abs(phi.at<float>((i+1),j+1))
                             && (phi.at<float>(i,j)>0) != (phi.at<float>((i+1),j+1)>0))
                        edg.at<float>(i,j) = fg;
                }
            }
        }
    }
   for (unsigned int i = 1; i < phi.rows-1; ++i)
    {
        for (unsigned int j = 1; j < phi.cols-1; ++j)
        {
            if (edg.at<float>((i-1),j-1)==fg
                && (phi.at<float>(i,j)>=0))
                edg.at<float>(i,j) = fg;
            else if (edg.at<float>((i-1),j)==fg
                     && (phi.at<float>(i,j)>=0))
                edg.at<float>(i,j) = fg;
            else if (edg.at<float>((i-1),j+1)==fg
                     && (phi.at<float>(i,j)>=0 ))
                edg.at<float>(i,j) = fg;
            else if (edg.at<float>(i,j-1)==fg
                     && (phi.at<float>(i,j)>=0))
                edg.at<float>(i,j) = fg;
            else if (edg.at<float>(i,j+1)==fg
                     && (phi.at<float>(i,j)>=0))
                edg.at<float>(i,j) = fg;
            else if (edg.at<float>((i+1),j-1)==fg
                     && (phi.at<float>(i,j)>=0))
                edg.at<float>(i,j) = fg;
            else if (edg.at<float>((i+1),j)==fg
                     && (phi.at<float>(i,j)>=0) )
                edg.at<float>(i,j) = fg;
            else if (edg.at<float>(i+1,j+1)==fg
                     && (phi.at<float>(i,j)>=0))
                edg.at<float>(i,j) = fg;
            
        }
    }
}



void ChanVeseImplementation::ChanVeseSegmentation(cv::Mat img)
{
    double P_ij;
    double deltaPhi;
    double F1;
    double F2;
    double F3;
    double F4;
    double F;
    double L;
    //double c1;
    //double c2;
    //CvMat* edges=cvCreateMat(img.rows,img.cols,CV_32FC1);
    // Segmentation parameters
    //double h = pCVinputs->h;
    double dt = pCVinputs->dt;
    double nu = pCVinputs->nu;
    double lambda1 = pCVinputs->lambda1;
    double lambda2 = pCVinputs->lambda2;
    unsigned int p = pCVinputs->p;
    
    // Variables to evaluate stopping condition
    for (int i=0;i<phi0.rows;i++)
        for (int j=0;j<phi0.cols;j++)
            phi.at<float>(i,j)=(float)phi0.at<float>(i,j);
    //NSLog(@"before: %f",phi.at<float>(193,158));
    pathEstimation.removerepetitivecoordinates(coOrdinates,phi);
    probabilities=pathEstimation.Laplacian();
    //return probabilities;
    //GetRegionAverages(img, h, c1, c2);
        if (1 == p)
        {
            L = 1.0;
        }
        else
        {
            L = 1.0; // fix this!!
        }

        // Loop through all interior image pixels
        for (int i = 1; i < (img.rows-1); i++)
        {
            for (int j = 1; j < (img.cols-1); j++)
            {
                // Compute coefficients needed in update
                GetChanVeseCoefficients(i, j,
                                        L,
                                        F1,
                                        F2,
                                        F3,
                                        F4,
                                        F,
                                        deltaPhi);
               double probabilitiesatiandj=probabilities[i][j];
                int k=8;
               /* P_ij = (float)phi.at<float>(i,j)
                - dt*deltaPhi*(nu +lambda1*(pow((double)(((2*((int)img.at<uchar>(i,j) -lowerThreshold))/(upperThreshold-lowerThreshold))-1),16))- lambda2*(-pow((double)((2*(int)img.at<uchar>(i,j)-(upperThreshold+lowerThreshold))/((upperThreshold-lowerThreshold))),16)+1)+(pow(2,-k/2)* pow(probabilitiesatiandj+0.4,(-(k/2)-1))*exp(-1/(2*(probabilitiesatiandj+0.4)))));*/
                P_ij = (float)phi.at<float>(i,j)
                - dt*deltaPhi*(nu +lambda1*(pow((((2*((int)img.at<uchar>(i,j) -lowerThreshold))/(upperThreshold-lowerThreshold))-1),16))-lambda2*(exp(-(pow(((2*(int)img.at<uchar>(i,j)-(upperThreshold+lowerThreshold))/(upperThreshold-lowerThreshold)),4))))+(pow(2,-k/2)*pow(probabilitiesatiandj+0.4,(-(k/2)-1))*exp(-1/(2*(probabilitiesatiandj+0.4)))));
                if(i==383 && j==330)
                {
                
                   // NSLog(@"%d",(int)img.at<uchar>(i,j));
                   // NSLog(@"%f",(pow((((2*((int)img.at<uchar>(i,j) -lowerThreshold))/(upperThreshold-lowerThreshold))-1),16)));
                   // NSLog(@"%f",(exp(-(pow(((2*(int)img.at<uchar>(i,j)-(upperThreshold+lowerThreshold))/(upperThreshold-lowerThreshold)),4)))));
                   // int ss=0;
                }
                /* P_ij = (float)phi.at<float>(i,j)
                - dt*deltaPhi*(nu + lambda1*pow((int)img.at<uchar>(i,j) -c1,2)
                              - lambda2*pow((int)img.at<uchar>(i,j) -c2,2));*/
                
                // Update level set function
                
                phi.at<float>(i,j)   = F1*(float)phi0.at<float>((i+1),j)
                + F2*(float)phi0.at<float>((i-1),j)
                + F3*(float)phi0.at<float>(i,j+1)
                + F4*(float)phi0.at<float>(i,j-1)
                + F*P_ij;
            }
        }
        // Update border values of phi by reflection
        for (int i = 0; i < img.rows; ++i)
        {
            phi.at<float>(i,0) = phi.at<float>(i,1);
            phi.at<float>(i,phi.cols-1)=phi.at<float>(i,phi.cols-2);
            // (*phi)->imageData[i*img->widthStep+img->width-1) = (*phi)->imageData[i*img->widthStep+img->width-2);
        }
        for (int j = 0; j < img.cols; ++j)
        {
            phi.at<float>(0,j)=phi.at<float>(1,j);
            phi.at<float>(((phi).rows-1),j)=phi.at<float>(((phi).rows-2),j);
        }
    ZeroCrossings((unsigned char)255,(unsigned char)0);
	ZeroCrossings2((unsigned char)255,(unsigned char)0);
    phi0=phi;
}

double ** ChanVeseImplementation::mainentrance(NSMutableArray *coordinates)
{
    int height;
    int width;
    coOrdinates=coordinates;
    height=sourceImage.rows;
    width=sourceImage.cols;
    phi0=cv::Mat(height,width,CV_32FC1);
    phi=cv::Mat(height,width,CV_32FC1);
    edg=cv::Mat(height,width,CV_32FC1);
    realedge=cv::Mat(height,width,CV_32FC1);
    blob=cv::Mat(height,width,CV_32FC1,float(0));
    pCVinputs=new struct CVsetup;
    pCVinputs->dt = 0.1;
    pCVinputs->h = 1.0;
    pCVinputs->lambda1 = 1.0;
    pCVinputs->lambda2 = 1.0;
    pCVinputs->mu = 0.5;
    pCVinputs->nu = 0;
    pCVinputs->p = 1;
    // Set up initial circular contour for a 256x256 image
    double x1;
    double y1;
    for (int i = 0; i < phi0.rows; i++)
    {
        for (int j = 0; j < phi0.cols; j++)
		{
            
			x1=double(i) - double(phi0.rows*1.0/4.0);
			y1 = double(j) - double(phi0.cols/2.0);
			phi0.at<float>(i,j)= (float)(64500.0/(64500.0 + x1*x1 + y1*y1) - 0.5);
           // phi0.at<float>(i,j)= (float)(10000.0/(10000.0 + x1*x1 + y1*y1) - 0.5);
            
		}
    }
    // the selection of lower and upper threshold from selected points starts here
    double maxi=-1000;
    double mini=1000;
    for(int i=0; i< coordinates.count;i++)
        {
            NSValue *value=[coordinates objectAtIndex:i];
            //NSLog(@"%f %f %d",(([value CGPointValue].x/768.0)*512.0),(([value CGPointValue].y)/2.0),(int)sourceImage.at<uchar>((int)(([value CGPointValue].y)/2.0),(int)(([value CGPointValue].x/768.0)*512.0)));
            if (sourceImage.at<uchar>([value CGPointValue].y,[value CGPointValue].x) > maxi)
            {
                maxi=sourceImage.at<uchar>([value CGPointValue].y,[value CGPointValue].x);
            }
            if (sourceImage.at<uchar>([value CGPointValue].y,[value CGPointValue].x) < mini)
            {
                mini=sourceImage.at<uchar>([value CGPointValue].y,[value CGPointValue].x);
            }
    }
    lowerThreshold=mini;
    upperThreshold=maxi;
    //NSLog(@"%f %f",lowerThreshold,upperThreshold);
    //lowerThreshold=1;
    //upperThreshold=60;
    
    // the selection of lower and upper threshold from selected points ends here
    
    pathEstimation.set(sourceImage,lowerThreshold,upperThreshold);
    pathEstimation.calcLaplacian();
    
    //Reading the csv file and exporting the content to an array starts here
    //NSString *file=@"pathprobabilityknee.csv";
    //NSString *file=@"pathprobability.csv";
    //NSString *file=@"pathprobabilitylung2.csv";
    //NSString *filepath=[[NSBundle mainBundle] pathForResource:file ofType:nil];
    //probabilities = [CSVParser parseCSVIntoArrayOfArraysFromFile:filepath
                                    //withSeparatedCharacterString:@","
                                      //      quoteCharacterString:nil];
    //Reading the csv file and exporting the content to an array ends here
    return 0;
}

/*
 //---------------------------------------------------------------------------//
 // function ReinitPhi
 // Reinitialize phi to the signed distance function to its zero contour
 void ReinitPhi(Image<double>* phiIn,
 Image<double>** psiOut,
 double dt,
 double h,
 unsigned int numIts)
 {
 if (0 == *psiOut)
 (*psiOut) = new Image<double>(phiIn->nRows(),phiIn->nCols());
 else if ((*psiOut)->nRows() != phiIn->nRows()
 || (*psiOut)->nCols() != phiIn->nCols())
 (*psiOut)->Allocate(phiIn->nRows(),phiIn->nCols());
 
 (*psiOut)->CopyFrom(phiIn);
 
 double a;
 double b;
 double c;
 double d;
 double x;
 double G;
 
 bool fStop = false;
 double Q;
 unsigned int M;
 Image<double>* psiOld = new Image<double>(phiIn->nRows(),phiIn->nCols());
 
 for (unsigned int k = 0; k < numIts && fStop == false; ++k)
 {
 psiOld->CopyFrom(*psiOut);
 for (unsigned int i = 1; i < phiIn->nRows()-1; ++i)
 {
 for (unsigned int j = 1; j < phiIn->nCols()-1; ++j)
 {
 a = (phiIn->data()[i)[j) - phiIn->data()[(i-1)*step+j))/h;
 b = (phiIn->data()[(i+1)*step+j) - phiIn->data()[i)[j))/h;
 c = (phiIn->data()[i)[j) - phiIn->data()[i)[j-1))/h;
 d = (phiIn->data()[i)[j+1) - phiIn->data()[i)[j))/h;
 
 if (phiIn->data()[i)[j) > 0)
 G = sqrt(max(max(a,0.0)*max(a,0.0),min(b,0.0)*min(b,0.0))
 + max(max(c,0.0)*max(c,0.0),min(d,0.0)*min(d,0.0))) - 1.0;
 else if (phiIn->data()[i)[j) < 0)
 G = sqrt(max(min(a,0.0)*min(a,0.0),max(b,0.0)*max(b,0.0))
 + max(min(c,0.0)*min(c,0.0),max(d,0.0)*max(d,0.0))) - 1.0;
 else
 G = 0;
 
 x = (phiIn->data()[i)[j) >= 0)?(1.0):(-1.0);
 (*psiOut)->data()[i)[j) = (*psiOut)->data()[i)[j) - dt*x*G;
 }
 }
 
 // Check stopping condition
 Q = 0;
 M = 0;
 for (unsigned int i = 0; i < phiIn->nRows(); ++i)
 {
 for (unsigned int j = 0; j < phiIn->nCols(); ++j)
 {
 if (abs(psiOld->data()[i)[j)) <= h)
 {
 ++M;
 Q += abs(psiOld->data()[i)[j) - (*psiOut)->data()[i)[j));
 }
 }
 }
 if (M != 0)
 Q = Q/((double)M);
 else
 Q = 0.0;
 
 if (Q < dt*h*h)
 {
 fStop = true;
 //cout << "Stopping condition reached at " << k+1 << " iterations; Q = " << Q << endl;
 }
 else
 {
 //cout << "Iteration " << k << ", Q = " << Q << " > " << dt*h*h << endl;
 }
 }
 }
 */
// Reinitialize phi to the signed distance function to its zero contour
// ReinitPhi(*phi, phi, 0.1, h, 100);
/*   if(k==0)
 {
 cvNamedWindow("image1", CV_WINDOW_AUTOSIZE);
 cvShowImage("image1", *phi);
 }
 if(k==1)
 {
 cvNamedWindow("image2", CV_WINDOW_AUTOSIZE);
 cvShowImage("image2", *phi);
 }
 if(k==2)
 {
 cvNamedWindow("image3", CV_WINDOW_AUTOSIZE);
 cvShowImage("image3", *phi);
 }
 if(k==3)
 {
 cvNamedWindow("image4", CV_WINDOW_AUTOSIZE);
 cvShowImage("image4", *phi);
 }
 if(k==4)
 {
 cvNamedWindow("image5", CV_WINDOW_AUTOSIZE);
 cvShowImage("image5", *phi);
 }*/

/*
 GaussianSmoothLabel=cv::Mat(height,width,CV_32FC1);
 for (int i = 0; i < sourceImage.rows; i++)
 {
 for (int j = 0; j < sourceImage.cols; j++)
 {
 GaussianSmoothLabel.at<float>(i,j)=0;
 }
 }
 
 for(int i=0; i< coordinates.count;i++)
 {
 NSValue *value=[coordinates objectAtIndex:i];
 //for(int i=-1;i<=1;i++)
 //  for(int j=-1;j<=1;j++)
 GaussianSmoothLabel.at<float>((int)(([value CGPointValue].y)/2.0),(int)(([value CGPointValue].x/768.0)*512.0))=1;
 }
 
 //blobdetection();
 cv::Mat bw=cv::Mat(height,width,CV_8UC1);
 for (int i = 0; i < phi0.rows; i++)
 {
 for (int j = 0; j < phi0.cols; j++)
 {
 bw.at<uchar>(i,j)=255;
 }
 }
 
 for (int i = 0; i < phi0.rows; i++)
 {
 for (int j = 0; j < phi0.cols; j++)
 {
 if(GaussianSmoothLabel.at<float>(i,j) > 0.0)
 bw.at<uchar>(i,j)=0.0;
 }
 }
 cv::Mat dist;
 cv::distanceTransform(bw, dist, CV_DIST_L2 , 5);
 
 mini=1000;
 maxi=-1000;
 for (int i = 0; i < phi0.rows; i++)
 {
 for (int j = 0; j < phi0.cols; j++)
 {
 if (dist.at<float>(i,j)< mini)
 {
 mini=dist.at<float>(i,j);
 }
 
 if (dist.at<float>(i,j) > maxi)
 {
 maxi=dist.at<float>(i,j);
 }
 }
 }
 
 for (int i = 0; i < phi0.rows; i++)
 {
 for (int j = 0; j < phi0.cols; j++)
 {
 dist.at<float>(i,j)=5*dist.at<float>(i,j)/maxi;
 }
 }
 
 float Miu=(upperThreshold+lowerThreshold)/2;
 
 //NSLog(@"%f %f",dist.at<float>(453,268),2/(1+pow(2*(((int)sourceImage.at<uchar>(453,268)-Miu)/(upperThreshold-lowerThreshold)), 16))-1);
 
 
 for (int i = 0; i < phi0.rows; i++)
 {
 for (int j = 0; j < phi0.cols; j++)
 {
 // if (GaussianSmoothLabel.at<float>(i,j)==0)
 //{
 GaussianSmoothLabel.at<float>(i,j)=2/(1+pow(2*(((int)sourceImage.at<uchar>(i,j)-Miu)/(upperThreshold-lowerThreshold)), 16))-1-dist.at<float>(i,j);
 //}
 }
 }
 */
//cv::GaussianBlur(GaussianSmoothLabel, GaussianSmoothLabel, cv::Size(5,5), 3);*/

