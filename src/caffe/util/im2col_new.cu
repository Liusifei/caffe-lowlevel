#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/util/device_alternate.hpp"
namespace caffe {

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im_ptr[i * width + j] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}




template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, height_col,
      width_col, data_col);
  //CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float* data_col);
template void im2col_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    double* data_col);


template <typename Dtype>
__global__ void im2col_v2_gpu_kernel(const int n, const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int out_w_offset = (index/channels) % width_col;
	int out_h_offset = ((index/channels)/width_col)%height_col;
	int c_im = index%channels;
	int temp_h_pad = out_h_offset * stride_h - pad_h;
	int temp_w_pad = out_w_offset * stride_w - pad_w;
	Dtype* data_col_ptr = data_col +  index  * kernel_h  *kernel_w;
	const Dtype* data_im_ptr = data_im + c_im*height *width;


	for(int h=0; h < kernel_h; ++h){
		for(int w = 0; w < kernel_w; ++w){
			int h_pad = temp_h_pad + h ;
			int w_pad = temp_w_pad + w;
			*data_col_ptr = (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) ?
					data_im_ptr[ h_pad *width+w_pad] : 0;
			data_col_ptr++;
		}
	}
  }
}




template <typename Dtype>
void im2col_v2_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_v2_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im,channels, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, height_col,
      width_col, data_col);
  //CUDA_POST_KERNEL_CHECK;
}

template void im2col_v2_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float* data_col);
template void im2col_v2_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    double* data_col);






template <typename Dtype>
__global__ void kernel_scale_im2col_gen_src_mat(const int nthreads, const Dtype* src,
	     Dtype* dest, const int height_col, const int width_col,const int channels,
	     const int new_kernel_h, const int new_kernel_w)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		int c = index% channels;
		int j = (index/channels)%width_col;
		int i = (index/channels)/width_col;
		int c_start = c*new_kernel_h * new_kernel_w;
		int c_interval = new_kernel_h * new_kernel_w;
		int buf_col_idx = (c_start*height_col+i)*width_col+j;
		int c_step = height_col*width_col;
		Dtype *m_ptr = dest+ (c+ channels*(j+i*width_col))*new_kernel_h*new_kernel_w;
		for(int cc = 0; cc< c_interval;++cc )
		{
			*m_ptr = (src[buf_col_idx+c_step*cc]);
			++m_ptr;
		}
	}
}

template <typename Dtype>
__global__ void kernel_scale_im2col_gen_dest_mat(const int nthreads, const Dtype* src,
	     Dtype* dest, const int height_col, const int width_col,const int channels,
	     const int kernel_h, const int kernel_w)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		int c = index% channels;
		int j = (index/channels)%width_col;
		int i = (index/channels)/width_col;
		int c_start = c*kernel_h * kernel_w;
		int c_interval = kernel_h * kernel_w;
		int buf_col_idx = (c_start*height_col+i)*width_col+j;
		int c_step = height_col*width_col;
		const Dtype *m_ptr = src+ (c+ channels*(j+i*width_col))*kernel_h*kernel_w;
		for(int cc = 0; cc< c_interval;++cc )
		{
			dest[buf_col_idx+c_step*cc] =(*m_ptr);
			++m_ptr;
		}
	}
}

template <typename Dtype>
void scale_im2col_gpu(const Dtype* data_im, const int channels,
	    const int height, const int width, const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w, const float cur_scale,
	    Dtype* data_col,Blob<Dtype>& blob_buf_col,Blob<Dtype>& blob_src_mat, Blob<Dtype>& blob_dest_mat)
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
		im2col_gpu(data_im,channels,height,width,new_kernel_h,new_kernel_w,pad_h+pad_h_to_add,
					  pad_w + pad_w_to_add,stride_h,stride_w,data_col);
		return;
	}

	blob_buf_col.Reshape(1,channels_col,height_col,width_col);
	Dtype * buf_col = blob_buf_col.mutable_gpu_data();
	im2col_gpu(data_im,channels,height,width,new_kernel_h,new_kernel_w,pad_h+pad_h_to_add,
			  pad_w + pad_w_to_add,stride_h,stride_w,buf_col);

	blob_src_mat.Reshape(1,channels*height_col*width_col,new_kernel_h, new_kernel_w);
	blob_dest_mat.Reshape(1,channels*height_col*width_col,kernel_h, kernel_w);

	kernel_scale_im2col_gen_src_mat<Dtype> <<<CAFFE_GET_BLOCKS(height_col*width_col*channels), CAFFE_CUDA_NUM_THREADS>>>(
			 height_col*width_col*channels, blob_buf_col.gpu_data(),
			 blob_src_mat.mutable_gpu_data(),  height_col, width_col, channels,
		      new_kernel_h, new_kernel_w);

	caffe::ResizeBlob_gpu(&blob_src_mat,&blob_dest_mat);

	kernel_scale_im2col_gen_dest_mat<Dtype> <<<CAFFE_GET_BLOCKS(height_col*width_col*channels), CAFFE_CUDA_NUM_THREADS>>>(
			 height_col*width_col*channels, blob_dest_mat.gpu_data(),
			 data_col,  height_col, width_col, channels,
		     kernel_h,  kernel_w);

	CUDA_POST_KERNEL_CHECK;


}

// added by alan
template void scale_im2col_gpu<float>(const float* data_im,const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h,
	const int stride_w, const float cur_scale, float* data_col,
	Blob<float>& blob_buf_col,Blob<float>& blob_src_mat, Blob<float>& blob_dest_mat);
template void scale_im2col_gpu<double>(const double* data_im,const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h,
	const int stride_w, const float cur_scale, double* data_col,
	Blob<double>& blob_buf_col,Blob<double>& blob_src_mat, Blob<double>& blob_dest_mat);



template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);
    /*
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        // the col location: [c * width * height + h_out, w_out]
        int c_col = c * patch_h * patch_w + (h - h_col * stride_h) * ksize
            + (w - w_col * stride_w);
        val += data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
    }
    */
    // equivalent implementation
    int offset =
        (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
    int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
    int coeff_w_col = (1 - stride_w * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im) {
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, patch_h, patch_w,
      pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im);
template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im);


template <typename Dtype>
__global__ void col2im_v2_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;

    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);

//    /*


    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
    	  const Dtype* data_col_ptr = data_col+( (h_col*width_col + w_col)*channels + c)*patch_h*patch_w;
    	  int h_in_patch = h - h_col*stride_h;
    	  int w_in_patch = w - w_col*stride_w;
    	  val+= data_col_ptr[ h_in_patch *patch_w + w_in_patch];

      }
    }
    data_im[index] = val;

  }
}


template <typename Dtype>
void col2im_v2_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im) {
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_v2_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, patch_h, patch_w,
      pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}
template void col2im_v2_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im);
template void col2im_v2_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im);


template <typename Dtype>
__global__ void kernel_scale_col2img_gen_src_mat(const int nthreads, const Dtype* src,
	     Dtype* dest, const int height_col, const int width_col,const int channels,
	     const int patch_h, const int patch_w)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		int c = index% channels;
		int j = (index/channels)%width_col;
		int i = (index/channels)/width_col;
		int c_start = c*patch_h * patch_w;
		int c_interval = patch_h*patch_w;
		int buf_col_idx = (c_start*height_col+i)*width_col+j;
		int c_step = height_col*width_col;
		Dtype *m_ptr = dest+ (c+ channels*(j+i*width_col))*patch_h*patch_w;
		for(int cc = 0; cc< c_interval;++cc )
		{
			*m_ptr = (src[buf_col_idx+c_step*cc]);
			++m_ptr;
		}
	}
}

template <typename Dtype>
__global__ void kernel_scale_col2img_gen_dest_mat(const int nthreads, const Dtype* src,
	     Dtype* dest, const int height_col, const int width_col,const int channels,
	     const int new_kernel_h, const int new_kernel_w)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		int c = index% channels;
		int j = (index/channels)%width_col;
		int i = (index/channels)/width_col;
		int c_start = c*new_kernel_h * new_kernel_w;
		int c_interval = new_kernel_h * new_kernel_w;
		int buf_col_idx = (c_start*height_col+i)*width_col+j;
		int c_step = height_col*width_col;
		const Dtype *m_ptr = src+ (c+ channels*(j+i*width_col))*new_kernel_h*new_kernel_w;
		for(int cc = 0; cc< c_interval;++cc )
		{
			dest[buf_col_idx+c_step*cc] =(*m_ptr);
			++m_ptr;
		}
	}
}


template<typename Dtype>
void scale_col2im_gpu(const Dtype* data_col, const int channels,
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
		col2im_gpu( data_col,channels,height, width, new_kernel_h, new_kernel_w,
					pad_h+pad_h_to_add, pad_w+pad_w_to_add,stride_h, stride_w,  data_im);
		return;
	}
	blob_buf_col.Reshape(1,channels_col,height_col,width_col);
//	Dtype * buf_col =blob_buf_col.mutable_gpu_data();

	blob_src_mat.Reshape(1,1,patch_h, patch_w);
	blob_dest_mat.Reshape(1,1,new_kernel_h,new_kernel_w);
//
//	Dtype * temp_col_source_mat = blob_src_mat.mutable_gpu_data();
//	Dtype *  temp_col_dest_mat = blob_dest_mat.mutable_gpu_data();
	kernel_scale_col2img_gen_src_mat<Dtype> <<<CAFFE_GET_BLOCKS(height_col*width_col*channels), CAFFE_CUDA_NUM_THREADS>>>(
				 height_col*width_col*channels, data_col,
				 blob_src_mat.mutable_gpu_data(),  height_col, width_col, channels,
			     patch_h, patch_w);

	caffe::ResizeBlob_gpu(&blob_src_mat,&blob_dest_mat);

	kernel_scale_im2col_gen_dest_mat<Dtype> <<<CAFFE_GET_BLOCKS(height_col*width_col*channels), CAFFE_CUDA_NUM_THREADS>>>(
				 height_col*width_col*channels, blob_dest_mat.gpu_data(),
				 blob_buf_col.mutable_gpu_data(),  height_col, width_col, channels,
			     new_kernel_h,  new_kernel_w);

	col2im_gpu( blob_buf_col.gpu_data(),channels,height, width, new_kernel_h, new_kernel_w,
			pad_h+pad_h_to_add, pad_w+pad_w_to_add,stride_h, stride_w,  data_im);
}

// added by alan
template void scale_col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const float cur_scale,float* data_im,
    Blob<float>& blob_buf_col,Blob<float>& blob_src_mat, Blob<float>& blob_dest_mat);
template void scale_col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const float cur_scale, double* data_im,
    Blob<double>& blob_buf_col,Blob<double>& blob_src_mat, Blob<double>& blob_dest_mat);

}  // namespace caffe
