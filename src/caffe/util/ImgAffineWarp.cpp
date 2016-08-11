#include "caffe/util/ImgAffineWarp.hpp"
using namespace std;
void mAffineWarp(const Mat_<float> M, const Mat& srcImg,Mat& dstImg,int interpolation)
{
    if(dstImg.empty())
        dstImg = Mat(srcImg.size(), srcImg.type());
    dstImg.setTo(0);

    for (int y=0; y<dstImg.rows; ++y)
    {
        for (int x=0; x<dstImg.cols; ++x)
        {
            float fx = M(0,0)*x + M(0,1)*y + M(0,2);
            float fy = M(1,0)*x + M(1,1)*y + M(1,2);

            int sy  = cvFloor(fy);
            int sx  = cvFloor(fx);
            fx -= sx;
            fy -= sy;
            
            if(sy<1 ||sy>srcImg.rows-2 || sx<1 || sx>srcImg.cols-2)
                continue;
            //sy = max(1, min(sy, srcImg.rows-2)); //my modify
            //sx = max(1, min(sx, srcImg.cols-2)); //my modify
            
            float w_y0 = abs(1.0f - fy);
            float w_y1 = abs(fy);
            float w_x0 = abs(1.0f-fx);
            float w_x1 = abs(fx);
            if(srcImg.channels()==1)
            {
                if(interpolation ==INTER_NEAREST)
                {
                    dstImg.at<uchar>(y, x) = srcImg.at<uchar>(sy, sx);
                }
                else
                {
                    dstImg.at<uchar>(y, x) = (srcImg.at<uchar>(sy, sx) * w_x0 * w_y0 + 
                            srcImg.at<uchar>(sy+1, sx) * w_x0 * w_y1 +
                            srcImg.at<uchar>(sy, sx+1) * w_x1 *w_y0 + 
                            srcImg.at<uchar>(sy+1, sx+1) * w_x1 * w_y1);
                }
            }
            else
            {
                if(interpolation ==INTER_NEAREST)
                {
                    for (int k=0; k<srcImg.channels(); ++k)
                        dstImg.at<cv::Vec3b>(y, x)[k] = srcImg.at<cv::Vec3b>(sy, sx)[k];
                }
                else
                {
                    for (int k=0; k<srcImg.channels(); ++k)
                    {
                        dstImg.at<cv::Vec3b>(y, x)[k] = (srcImg.at<cv::Vec3b>(sy, sx)[k] * w_x0 * w_y0 +
                                srcImg.at<cv::Vec3b>(sy+1, sx)[k] * w_x0 * w_y1 +
                                srcImg.at<cv::Vec3b>(sy, sx+1)[k] * w_x1 *w_y0 +
                                srcImg.at<cv::Vec3b>(sy+1, sx+1)[k] * w_x1 * w_y1);
                    }
                }
            }
        }
    }
}
