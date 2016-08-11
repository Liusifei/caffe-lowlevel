#ifndef _CAFFE_UTIL_IM2COL_HPP_
#define _CAFFE_UTIL_IM2COL_HPP_
#include "caffe/blob.hpp"
namespace caffe {

/**
 * col_buffer_.Reshape(
      1, channels_ * kernel_h_ * kernel_w_, height_out_, width_out_);
 */
template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_col);

/**
 * col_buffer_.Reshape(
     1, channels_ * kernel_h_ * kernel_w_, n_actived_count,1);
 */
template <typename Dtype>
void sparse_im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,const int stride_w,
    Dtype* data_col,const int n_actived_count, int* height_col_arr, int* width_col_arr);



/**
 * data_col_sparse_.Reshape(
     1, num_output_, n_actived_count,1);
 * data_col_full.Reshape(
      1, num_output_, height_out_, width_out_);
 */

template <typename Dtype>
void sparse_col2full_cpu(const Dtype* data_col_sparse,
		const int n_actived_count, const int* height_col_arr,const  int* width_col_arr,
		Dtype* data_col_full,const int num_output, const int height_out, const int width_out);




template <typename Dtype>
void get_sparse_mask(const Dtype* data_im,const int height, const int width,
		const int kernel_h, const int kernel_w,const int stride_h,const int stride_w,
		const int pad_h, const int pad_w,int* res_height_arr, int* res_width_arr,int* size_res);


/*  out: Reshape(1, channels_ * height_out_ * width_out_, kernel_h, kernel_w)
 *
 *
 * */

template <typename Dtype>
void im2col_v2_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_col);

template <typename Dtype>
void im2col_v2_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_col);




template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im);


/*  in: Reshape(1, channels_ * height_out_ * width_out_, kernel_h, kernel_w)
 *
 * */
template <typename Dtype>
void col2im_v2_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im);

template <typename Dtype>
void col2im_v2_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im);

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_col);

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im);


// added by alan
template<typename Dtype>
void get_patch_from_im2col(const Dtype * data_col, const int channels,
	    const int height, const int width, const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w,const int col_c_idx,const int col_h_idx,const int col_w_idx,
	    Dtype* res_img);

template<typename Dtype>
void get_patch_from_im2col_v2(const Dtype * data_col, const int channels,
	    const int height, const int width, const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w,const int col_c_idx,const int col_h_idx,const int col_w_idx,
	    Dtype* res_img);


template <typename Dtype>
void generate_sample_img(const int channels, const int height, const int width,Dtype * data_res);

template <typename Dtype>
void scale_im2col_cpu(const Dtype* data_im, const int channels,
	    const int height, const int width, const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w, const float cur_scale,
	    Dtype* data_col, Blob<Dtype>& blob_buf_col,Blob<Dtype>& blob_src_mat, Blob<Dtype>& blob_dest_mat);

template<typename Dtype>
void scale_col2im_cpu(const Dtype* data_col, const int channels,
	    const int height, const int width, const int patch_h, const int patch_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w,const float cur_scale,
	    Dtype* data_im, Blob<Dtype>& blob_buf_col,Blob<Dtype>& blob_src_mat, Blob<Dtype>& blob_dest_mat);

template <typename Dtype>
void scale_im2col_gpu(const Dtype* data_im, const int channels,
	    const int height, const int width, const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w, const float cur_scale,
	    Dtype* data_col, Blob<Dtype>& blob_buf_col,Blob<Dtype>& blob_src_mat, Blob<Dtype>& blob_dest_mat);

template<typename Dtype>
void scale_col2im_gpu(const Dtype* data_col, const int channels,
	    const int height, const int width, const int patch_h, const int patch_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w,const float cur_scale,
	    Dtype* data_im, Blob<Dtype>& blob_buf_col,Blob<Dtype>& blob_src_mat, Blob<Dtype>& blob_dest_mat);

}  // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
