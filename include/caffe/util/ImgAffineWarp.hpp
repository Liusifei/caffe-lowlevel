#pragma once
#include <vector>
#include <opencv/cv.h>
using namespace std;
using namespace cv;
#define SQR(a) ((a)*(a))

template<typename T>  inline Mat_<float> Get_Affine_matrix(T &srcCenter, T &dstCenter,float alpha, float scale)
{
	Mat_<float> M(2,3);
	M(0,0) = scale*cos(alpha);
	M(0,1) = scale*sin(alpha);
	M(1,0) = -M(0,1);
	M(1,1) =  M(0,0);

	M(0,2) = srcCenter.x - M(0,0)*dstCenter.x - M(0,1)*dstCenter.y;
	M(1,2) = srcCenter.y - M(1,0)*dstCenter.x - M(1,1)*dstCenter.y;
	return M;
}
inline Mat_<float> inverseMatrix(Mat_<float>& M)
{
	double D = M(0,0)*M(1,1) - M(0,1)*M(1,0);
	D = D != 0 ? 1./D : 0;

	Mat_<float> inv_M(2,3);

	inv_M(0,0) = M(1,1)*D;
	inv_M(0,1) = M(0,1)*(-D);
	inv_M(1,0) = M(1,0)*(-D);
	inv_M(1,1) = M(0,0)*D;

	inv_M(0,2) = -inv_M(0,0)*M(0,2) - inv_M(0,1)*M(1,2);
	inv_M(1,2) = -inv_M(1,0)*M(0,2) - inv_M(1,1)*M(1,2);
	return inv_M;
}
void mAffineWarp(const Mat_<float> M, const Mat& srcImg,Mat& dstImg,int interpolation=INTER_LINEAR);
template<typename T>  inline void Affine_Point(const Mat_<float> &M,T& srcPt, T &dstPt)
{
    float x = M(0,0)*srcPt.x + M(0,1)*srcPt.y + M(0,2);
    float y = M(1,0)*srcPt.x + M(1,1)*srcPt.y + M(1,2);
    dstPt.x = x;
    dstPt.y = y;
}