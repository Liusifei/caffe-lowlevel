#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/util_img.hpp"

namespace caffe {

/*  out: Reshape(1* height_out_ * width_out_, channels_ , kernel_h, kernel_w)
 * */
template <typename Dtype>
void im2col_v2_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_col){

	int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	int channels_col = height_col * width_col;
	for(int c= 0 ; c < channels_col; ++c){
		int out_w_offset = c % width_col;
		int out_h_offset = (c/width_col)%height_col;
		//int c_im = c / width_col / height_col;
		for(int c_im = 0; c_im < channels; c_im++)
		{
			for(int h=0; h < kernel_h; ++h){
				for(int w = 0; w < kernel_w; ++w){
					int h_pad = out_h_offset * stride_h - pad_h + h ;
					int w_pad = out_w_offset * stride_w - pad_w + w;
					data_col[((c*channels + c_im) * kernel_h + h)*kernel_w + w ] = (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) ?
							data_im[(c_im*height +h_pad)*width+w_pad] : 0;
				}
			}
		}
	}
}


/**
 * height_col_arr should be equal to h * 1
 * similar as width_col_arr
 * h * 1 record the  coordinate for each output element
 */
template <typename Dtype>
void sparse_im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,const int stride_h,const int stride_w, Dtype* data_col,
    const int n_actived_count, int* height_col_arr, int* width_col_arr){

	int channels_col = channels * kernel_h * kernel_w;
	for (int c = 0; c < channels_col; ++c) {
	    int w_offset = c % kernel_w;
	    int h_offset = (c / kernel_w) % kernel_h;
	    int c_im = c / kernel_h / kernel_w;
	    for(int actived_idx = 0; actived_idx < n_actived_count; ++actived_idx  )
	    {
	    	int h_pad = height_col_arr[actived_idx]*stride_h - pad_h + h_offset;
	    	int w_pad = width_col_arr[actived_idx]*stride_w - pad_w + w_offset;
	    	if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
	    	{
	    		data_col[c* n_actived_count + actived_idx] =
	    		            data_im[(c_im * height + h_pad) * width + w_pad];
	    	}else
	    	{
	    		data_col[c* n_actived_count + actived_idx] = 0;
	    	}
	    }
	}
}


template void sparse_im2col_cpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,const int stride_h,const int stride_w, float* data_col,
    const int n_actived_count, int* height_col_arr, int* width_col_arr);

template void sparse_im2col_cpu(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,const int stride_h,const int stride_w, double* data_col,
    const int n_actived_count, int* height_col_arr, int* width_col_arr);


/**
 * height_col_arr should be equal to h * 1
 * similar as width_col_arr
 * h * 1 record the  coordinate for each output element
 */
template <typename Dtype>
void get_sparse_mask(const Dtype* data_in,const int height, const int width,
		const int kernel_h, const int kernel_w,const int stride_h,const int stride_w,
		const int pad_h, const int pad_w, int* res_height_arr, int* res_width_arr,int* size_res){
	size_res[0] = 0;
	for(int h = 0; h < height; h+= stride_h)
	{
		if(h + kernel_h <= height+pad_h)
		{
			for(int w = 0; w < width; w+= stride_w)
			{
				if(data_in[h*width + w] == Dtype(1) && w + kernel_w <= width+pad_w )
				{
					res_width_arr[size_res[0]] = w/stride_w;
					res_height_arr[size_res[0]] = h/stride_h;
					++size_res[0];
				}
			}
		}
	}
}


template void get_sparse_mask (const float* data_im,const int height, const int width,
		const int kernel_h, const int kernel_w,const int stride_h,const int stride_w,
		const int pad_h, const int pad_w, int* res_height_arr, int* res_width_arr,int* size_res);
template void get_sparse_mask (const double* data_im,const int height, const int width,
		const int kernel_h, const int kernel_w,const int stride_h,const int stride_w,
		const int pad_h, const int pad_w, int* res_height_arr, int* res_width_arr,int* size_res);


/**
 * data_col_sparse_.Reshape(
     1, num_output_, n_actived_count,1);
 * data_col_full.Reshape(
      1, num_output_, height_out_, width_out_);
 */

template <typename Dtype>
void sparse_col2full_cpu(const Dtype* data_col_sparse,
		const int n_actived_count, const int* height_col_arr,const  int* width_col_arr,
		Dtype* data_col_full,const int num_output, const int height_out, const int width_out){
	int channels_col = n_actived_count*num_output;
	for(int i = 0; i < channels_col; ++i)
	{
		int full_channel_id =  i/n_actived_count ;
		int full_idx_h = height_col_arr[i%n_actived_count];
		int full_idx_w = width_col_arr[i%n_actived_count];
		int idx = (full_channel_id*height_out  +  full_idx_h )* width_out + full_idx_w;
		data_col_full[idx] = data_col_sparse[i];
	}

}


template void sparse_col2full_cpu(const float* data_col_sparse,
		const int n_actived_count, const int* height_col_arr,const  int* width_col_arr,
		float* data_col_full,const int num_output, const int height_out, const int width_out);
template void sparse_col2full_cpu(const double* data_col_sparse,
		const int n_actived_count, const int* height_col_arr,const  int* width_col_arr,
		double* data_col_full,const int num_output, const int height_out, const int width_out);



template <typename Dtype>
/**
 *bit -> row
 *multi row  -> to a map in one channel
 *multi channel -> data_col
 *
 *[im_channel][im_h][im_w]
 *
 */
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col) {
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
            data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

template<typename Dtype>
void get_patch_from_im2col_v2(const Dtype * data_col, const int channels,
	    const int height, const int width, const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w,const int col_c_idx,const int col_h_idx,const int col_w_idx,
	    Dtype* res_img){

	  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

	  if(col_h_idx < 0 || col_w_idx<0  || col_c_idx < 0  )
	  {
		  printf("invalid col_h_idx or col_w_idx or col_c_idx , can not be 0 \n");
		  return;
	  }

	  if(height_col <= col_h_idx || width_col <= col_w_idx || col_c_idx >= channels)
	  {
		  printf("invalid col_h_idx or col_w_idx or col_c_idx\n");
		  return;
	  }
	  int c_start = col_c_idx*kernel_h * kernel_w;
	  int c_end = (col_c_idx+1)*kernel_h * kernel_w;
	  for(int idx =c_start;idx<c_end;idx++)
	  {
		  int buf_col_idx = kernel_w*kernel_h*(  (col_c_idx*height_col + col_h_idx) * width_col +col_w_idx) + (idx - c_start);
		  res_img[idx - c_start] = data_col[buf_col_idx];
	  }
}

template void get_patch_from_im2col_v2(const float * data_col, const int channels,
	    const int height, const int width, const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w,const int col_c_idx,const int col_h_idx,const int col_w_idx,
	    float* res_img);


template void get_patch_from_im2col_v2(const double * data_col, const int channels,
	    const int height, const int width, const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w,const int col_c_idx,const int col_h_idx,const int col_w_idx,
	    double* res_img);

// get a         
template<typename Dtype>
void get_patch_from_im2col(const Dtype* data_col, const int channels,
	    const int height, const int width, const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w,const int col_c_idx,const int col_h_idx,const int col_w_idx,
	    Dtype* res_img)
{
	  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

	  if(col_h_idx < 0 || col_w_idx<0  || col_c_idx < 0  )
	  {
		  printf("invalid col_h_idx or col_w_idx or col_c_idx , can not be 0 \n");
		  return;
	  }

	  if(height_col <= col_h_idx || width_col <= col_w_idx || col_c_idx >= channels)
	  {
		  printf("invalid col_h_idx or col_w_idx or col_c_idx\n");
		  return;
	  }
	  int c_start = col_c_idx*kernel_h * kernel_w;
	  int c_end = (col_c_idx+1)*kernel_h * kernel_w;
	  for(int idx =c_start;idx<c_end;idx++)
	  {
		  int buf_col_idx = (idx*height_col+col_h_idx)*width_col+col_w_idx;
		  res_img[idx - c_start] = data_col[buf_col_idx];
	  }
}
template <typename Dtype>
void generate_sample_img(const int channels, const int height, const int width,Dtype * data_res)
{
	for(int i=0;i<channels*height*width;++i)
	{
		data_res[i] = i;
	}
}

// an implementation of im2col_cpu adding scale transformation

//void print_cvmat(const cv::Mat& mat)
//{
//	for(int h=0;h<mat.rows;h++)
//	{
//		for(int w=0;w<mat.cols;w++)
//		{
//			printf("%6.2lf ",(*(double*)mat.ptr(h,w)));
//		}
//		printf("\n");
//	}
//}

// cur_scale must be positive.
template <typename Dtype>
void scale_im2col_cpu(const Dtype* data_im, const int channels,
	    const int height, const int width, const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w, const float cur_scale,
	    Dtype* data_col, Blob<Dtype>& blob_buf_col,Blob<Dtype>& blob_src_mat, Blob<Dtype>& blob_dest_mat)
{
	// calculate the new pad and kernel information
	int pad_h_to_add = (floor(kernel_h*(pow(2,cur_scale)-1)))/2;
	int pad_w_to_add = (floor(kernel_w*(pow(2,cur_scale)-1)))/2;
	int new_kernel_h = pad_h_to_add*2 + kernel_h;
	int new_kernel_w = pad_w_to_add*2 + kernel_w;
	int height_col = (height + 2 * (pad_h + pad_h_to_add) - new_kernel_h) / stride_h + 1;
	int width_col = (width + 2 * (pad_w + pad_h_to_add) - new_kernel_w) / stride_w + 1;
	int channels_col = channels * new_kernel_h * new_kernel_w;

	if((new_kernel_h == kernel_h) &&(new_kernel_w ==  kernel_w))
	{
		im2col_cpu(data_im,channels,height,width,new_kernel_h,new_kernel_w,pad_h+pad_h_to_add,
					  pad_w + pad_w_to_add,stride_h,stride_w,data_col);
		return;
	}

//	printf("pad_h = %d     pad_w = %d    kernel_h = %d    kernel_w = %d  scale = %f \n",
//			pad_h,pad_w,kernel_h,kernel_w,cur_scale);
//	printf("new_pad_h = %d  new_pad_w = %d  new_kernel_h = %d  new_kernel_w = %d \n",
//			pad_h_to_add+pad_h, pad_w+pad_w_to_add,new_kernel_h, new_kernel_w);

	blob_buf_col.Reshape(1,channels_col,height_col,width_col);
	Dtype * buf_col = blob_buf_col.mutable_cpu_data();

	// im2col for new parameters
	im2col_cpu(data_im,channels,height,width,new_kernel_h,new_kernel_w,pad_h+pad_h_to_add,
			  pad_w + pad_w_to_add,stride_h,stride_w,buf_col);
	//[a_chanel][b_height][c_width] = (a*height+b)*width + c
	//cv::Mat temp_col_mat = cvCreateMat(new_kernel_h,new_kernel_w,CV_64FC1);
	//cv::Mat temp_col_dest_mat  =  cvCreateMat(kernel_h,kernel_w,CV_64FC1);



	blob_src_mat.Reshape(1,channels*height_col*width_col,new_kernel_h, new_kernel_w);
	blob_dest_mat.Reshape(1,channels*height_col*width_col,kernel_h, kernel_w);
	Dtype *temp_col_mat = blob_src_mat.mutable_cpu_data();
	Dtype * temp_col_dest_mat = blob_dest_mat.mutable_cpu_data();

	for(int i=0;i<height_col;i++)
	{
		for(int j=0;j<width_col;j++)
		{
			for(int c=0; c<channels;c++)
			{
				int c_start = c*new_kernel_h * new_kernel_w;
				int c_end = (c+1)*new_kernel_h * new_kernel_w;
				int buf_col_idx = (c_start*height_col+i)*width_col+j;
				int c_step = height_col*width_col;
				Dtype *m_ptr =temp_col_mat+blob_src_mat.offset(0,c+ channels*(j+i*width_col));
				for(int cc = c_start; cc<c_end;cc++ )
				{
					*m_ptr = (buf_col[buf_col_idx]);
					++m_ptr;
					buf_col_idx += c_step;
				}
			}
		}
	}

	//std::cout<<"before resize"<<std::endl;
	caffe::ResizeBlob_cpu(&blob_src_mat,&blob_dest_mat);
	//std::cout<<"after resize"<<std::endl;
	for(int i=0;i<height_col;i++)
	{
		for(int j=0;j<width_col;j++)
		{
			for(int c=0; c<channels;c++)
			{
				int c_start = c*kernel_h*kernel_w;
				int c_end = (c+1)*kernel_h*kernel_w;
				int buf_col_idx = (c_start*height_col+i)*width_col+j;
				int c_step = height_col*width_col;
				Dtype* m_ptr =temp_col_dest_mat+blob_dest_mat.offset(0,c+ channels*(j+i*width_col));
				for(int cc=c_start;cc<c_end;cc++)
				{
					data_col[buf_col_idx] =(*m_ptr);
					++m_ptr;
					buf_col_idx += c_step;
				}
			}
		}
	}

//
//	for(int i=0;i<height_col;i++)
//	{
//		for(int j=0;j<width_col;j++)
//		{
//			for(int c=0; c<channels;c++)
//			{
//				int c_start = c*new_kernel_h * new_kernel_w;
//				int c_end = (c+1)*new_kernel_h * new_kernel_w;
//				int buf_col_idx = (c_start*height_col+i)*width_col+j;
//				int c_step = height_col*width_col;
//				Dtype *m_ptr =temp_col_mat;
//				for(int cc = c_start; cc<c_end;cc++ )
//				{
//					*m_ptr = (buf_col[buf_col_idx]);
//					++m_ptr;
//					buf_col_idx += c_step;
//				}
//
//				caffe::BiLinearResizeMat_cpu(temp_col_mat,new_kernel_h, new_kernel_w,temp_col_dest_mat,kernel_h,kernel_w);
//
//				c_start = c*kernel_h*kernel_w;
//				c_end = (c+1)*kernel_h*kernel_w;
//				buf_col_idx = (c_start*height_col+i)*width_col+j;
//				m_ptr =temp_col_dest_mat;
//				for(int cc=c_start;cc<c_end;cc++)
//				{
//					data_col[buf_col_idx] =(*m_ptr);
//					++m_ptr;
//					buf_col_idx += c_step;
//				}
//			}
//		}
//	}


}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_col);


template void im2col_v2_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_col);
template void im2col_v2_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_col);


// added by alan
template void scale_im2col_cpu<float>(const float* data_im,const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h,
	const int stride_w, const float cur_scale, float* data_col,
	Blob<float>& blob_buf_col,Blob<float>& blob_src_mat, Blob<float>& blob_dest_mat);
template void scale_im2col_cpu<double>(const double* data_im,const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h,
	const int stride_w, const float cur_scale, double* data_col,
	Blob<double>& blob_buf_col,Blob<double>& blob_src_mat, Blob<double>& blob_dest_mat);

template void generate_sample_img<double>(const int channels,
		const int height, const int width,double * data_res);
template void get_patch_from_im2col(const double* data_col, const int channels,
	    const int height, const int width, const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w,const int col_c_idx,const int col_h_idx,const int col_w_idx,
	    double* res_img
	    );
template void get_patch_from_im2col(const float* data_col, const int channels,
	    const int height, const int width, const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w,const int col_c_idx,const int col_h_idx,const int col_w_idx,
	    float* res_img );


/*  in: Reshape(1* height_out_ * width_out_, channels_ , kernel_h, kernel_w)
 *
 * */
template <typename Dtype>
void col2im_v2_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im){
	caffe_set(height * width * channels, Dtype(0), data_im);
	int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
	int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
	int channels_col = height_col * width_col;
	for(int c= 0 ; c < channels_col; ++c){
		int col_w_offset = c % width_col;
		int col_h_offset = (c/width_col)%height_col;

		for(int c_im =0 ; c_im < channels; ++c_im)
		{
			for(int h=0; h < patch_h; ++h){
				for(int w = 0; w < patch_w; ++w){
					int h_pad = col_h_offset * stride_h - pad_h + h ;
					int w_pad = col_w_offset * stride_w - pad_w + w;
					data_im[(c_im*height +h_pad)*width+w_pad] += (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)?
							data_col[((c*channels+c_im) * patch_h + h)*patch_w + w ] :0;
				}
			}
		}
	}
}
template void col2im_v2_cpu(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im);
template void col2im_v2_cpu(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im);

/**
 * col_buffer_.Reshape(
      1, channels_ * kernel_h_ * kernel_w_, height_out_, width_out_);
 */
template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_im) {
  caffe_set(height * width * channels, Dtype(0), data_im);
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int channels_col = channels * patch_h * patch_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % patch_w;
    int h_offset = (c / patch_w) % patch_h;
    int c_im = c / patch_h / patch_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
              data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}

template<typename Dtype>
void scale_col2im_cpu(const Dtype* data_col, const int channels,
	    const int height, const int width, const int patch_h, const int patch_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w,const float cur_scale,
	    Dtype* data_im,Blob<Dtype>& blob_buf_col,Blob<Dtype>& blob_src_mat, Blob<Dtype>& blob_dest_mat)
{
	// calculate the new pad and kernel information
	int pad_h_to_add = (floor(patch_h*(pow(2,cur_scale)-1)))/2;
	int pad_w_to_add = (floor(patch_w*(pow(2,cur_scale)-1)))/2;
	int new_kernel_h = pad_h_to_add*2 + patch_h;
	int new_kernel_w = pad_w_to_add*2 + patch_w;
	int height_col = (height + 2 * (pad_h + pad_h_to_add) - new_kernel_h) / stride_h + 1;
	int width_col = (width + 2 * (pad_w + pad_h_to_add) - new_kernel_w) / stride_w + 1;
	int channels_col = channels * new_kernel_h * new_kernel_w;


	if((new_kernel_h == patch_h) && (new_kernel_w == patch_w))
	{
		col2im_cpu( data_col,channels,height, width, new_kernel_h, new_kernel_w,
					pad_h+pad_h_to_add, pad_w+pad_w_to_add,stride_h, stride_w,  data_im);
		return;
	}
	blob_buf_col.Reshape(1,channels_col,height_col,width_col);
	Dtype * buf_col =blob_buf_col.mutable_cpu_data();
//	cv::Mat temp_col_source_mat = cvCreateMat(patch_h,patch_w,CV_64FC1);
//	cv::Mat temp_col_dest_mat  =  cvCreateMat(new_kernel_h,new_kernel_w,CV_64FC1);
	blob_src_mat.Reshape(1,1,patch_h, patch_w);
	blob_dest_mat.Reshape(1,1,new_kernel_h,new_kernel_w);

	Dtype * temp_col_source_mat = blob_src_mat.mutable_cpu_data();
	Dtype *  temp_col_dest_mat = blob_dest_mat.mutable_cpu_data();

//	printf("pad_h = %d     pad_w = %d    kernel_h = %d    kernel_w = %d  scale = %f \n",
//			pad_h,pad_w,patch_h,patch_w,cur_scale);
//	printf("new_pad_h = %d  new_pad_w = %d  new_kernel_h = %d  new_kernel_w = %d \n",
//			pad_h_to_add+pad_h, pad_w+pad_w_to_add,new_kernel_h, new_kernel_w);
//

	for(int i = 0;i<height_col;++i)
	{
		for(int j=0;j<width_col;j++)
		{
			for(int c=0; c<channels;c++)
			{
				int c_start = c*patch_h*patch_w;
				int c_end = (c+1)*patch_h*patch_w;
				int buf_col_idx = (c_start*height_col+i)*width_col+j;
				int c_step = height_col*width_col;
				Dtype *m_ptr = temp_col_source_mat ;
				for(int cc = c_start; cc<c_end;cc++ )
				{
					*m_ptr =  (data_col[buf_col_idx]);
					++m_ptr;
					buf_col_idx += c_step;
				}


				//cv::resize(temp_col_source_mat,temp_col_dest_mat,cv::Size(new_kernel_h,new_kernel_w),cv::INTER_LINEAR);
				caffe::BiLinearResizeMat_cpu(temp_col_source_mat,patch_h,patch_w,temp_col_dest_mat,new_kernel_h,new_kernel_w);
//				printf("the small pathes: \n");
//				print_cvmat(temp_col_source_mat);
//				printf("the resized patch: \n");
//				print_cvmat(temp_col_dest_mat);
//				printf("\n\n");
//				Mypause();
				c_start = c*new_kernel_h*new_kernel_w;
				c_end = (c+1)*new_kernel_h*new_kernel_w;
				buf_col_idx = (c_start*height_col+i)*width_col+j;
				m_ptr = temp_col_dest_mat ;
				for(int cc=c_start;cc<c_end;cc++)
				{
					buf_col[buf_col_idx] = (*m_ptr);
					++m_ptr;
					buf_col_idx += c_step;
				}

			}
		}
	}

	col2im_cpu( buf_col,channels,height, width, new_kernel_h, new_kernel_w,
			pad_h+pad_h_to_add, pad_w+pad_w_to_add,stride_h, stride_w,  data_im);
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im);



// added by alan
template void scale_col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const float cur_scale,float* data_im,
    Blob<float>& blob_buf_col,Blob<float>& blob_src_mat, Blob<float>& blob_dest_mat);
template void scale_col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const float cur_scale, double* data_im,
    Blob<double>& blob_buf_col,Blob<double>& blob_src_mat, Blob<double>& blob_dest_mat);
}  // namespace caffe
