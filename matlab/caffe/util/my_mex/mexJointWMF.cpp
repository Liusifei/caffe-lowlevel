#include <math.h>
#include <matrix.h>
#include <mex.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include "opencv2/core/core.hpp"
#include <time.h>

#include "jointWMF.h"

using namespace std;
using namespace cv;

Mat mxArray2mat(const mxArray *data){ // CAN covert uchar/double any # channels mxArray to mat

    const int *dims = mxGetDimensions(data);
    int n = mxGetNumberOfDimensions(data);
    int rows = dims[0];
    int cols = dims[1];
    int chs = (n==3)?dims[2]:1;
    int rows_cols = rows*cols;
    
    Mat ret;

    if (mxIsUint8(data)){
      ret = Mat(rows, cols, CV_MAKETYPE(CV_8U,chs));
        
      uchar *imgData = (uchar *)mxGetPr(data);

      for (int i=0; i<rows; i++){
          for (int j=0; j<cols; j++){
              for(int k=0; k<chs; k++){
                ret.ptr<uchar>(i,j)[k] = imgData[k*rows_cols+j*rows+i];
              }
          }
       }
    }
    else if (mxIsDouble(data)){
      ret = Mat(rows, cols, CV_MAKETYPE(CV_32F,chs));
      double * imgData = (double *)mxGetPr(data);

      for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
          for(int k=0; k<chs; k++){
            ret.ptr<float>(i,j)[k] = (float)imgData[k*rows_cols+j*rows+i];
          }
        }
      }
    }

    return ret;
}


mxArray* mat2mxArray(Mat& m) // Only output double mxArray of any # channels
{   
   int rows=m.rows;
   int cols=m.cols;
   int rows_cols = rows*cols;
   int chs=m.channels();
   mwSize ndim = 3;
   mwSize dims[3] = {rows,cols,chs};
   //Mat data is float, and mxArray uses double, so we need to convert.   
   mxArray *T=mxCreateNumericArray (ndim, dims, mxDOUBLE_CLASS, mxREAL);
   double *buffer=(double*)mxGetPr(T);
   for(int i=0; i<rows; i++){
       for(int j=0; j<cols; j++){
          for(int k=0;k<chs; k++){
            buffer[k*rows_cols+j*rows+i]= (double)m.ptr<float>(i, j)[k];
         }
       }
   }

   return T;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	Mat img = mxArray2mat(prhs[0]); // must exist
  Mat feature = mxArray2mat(prhs[1]); // must exist
  int r = 10;
  float sigma = 25.5;
  int nI = 256;
  int nF = 256;
  int iter = 1;
  Mat mask = Mat();
  string weightType = "exp";
  
  if(nrhs>=3) r = (int)(((double *)mxGetPr(prhs[2]))[0]+0.5);
  if(nrhs>=4) sigma = (float)(((double *)mxGetPr(prhs[3]))[0]);
  if(nrhs>=5) nI = (int)(((double *)mxGetPr(prhs[4]))[0]+0.5);
  if(nrhs>=6) nF = (int)(((double *)mxGetPr(prhs[5]))[0]+0.5);
  if(nrhs>=7) iter = (int)(((double *)mxGetPr(prhs[6]))[0]+0.5);
  if(nrhs>=8) {
    mxChar* str = mxGetChars(prhs[7]);
    char tmp[10]={0};
    for(int i=0;i<3;i++)tmp[i]=(char)str[i];

    weightType = tmp;
  }
  if(nrhs>=9){
    mask = mxArray2mat(prhs[8]);
  }

	Mat res1 = JointWMF::filter(img,feature,r,sigma,nI,nF,iter,weightType,mask);

	res1.convertTo(res1,CV_32F);

	if(nlhs>0){
		plhs[0] = mat2mxArray(res1);
	}
}