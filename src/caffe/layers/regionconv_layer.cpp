#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
void RegionconvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      
      CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
          << "corresponding to (num, channels, height, width)";
      // Configure the kernel size, padding, stride, and inputs.
      RegionconvolutionParameter conv_param = this->layer_param_.regionconvolution_param();
      
      CHECK(!conv_param.has_input_patch_size() !=
          !(conv_param.has_input_patch_h() && conv_param.has_input_patch_w()))
          << "Filter size is patch_size OR patch_h and patch_w; not both";
      CHECK(conv_param.has_input_patch_size() ||
          (conv_param.has_input_patch_h() && conv_param.has_input_patch_w()))
          << "For non-square filters both patch_h and patch_w are required.";
          
      CHECK(!conv_param.has_output_patch_size() !=
          !(conv_param.has_output_patch_h() && conv_param.has_output_patch_w()))
          << "Filter size is patch_size OR patch_h and patch_w; not both";
      CHECK(conv_param.has_output_patch_size() ||
          (conv_param.has_output_patch_h() && conv_param.has_output_patch_w()))
          << "For non-square filters both patch_h and patch_w are required.";
          
      CHECK((!conv_param.has_input_pad() && conv_param.has_input_pad_h()
          && conv_param.has_input_pad_w())
          || (!conv_param.has_input_pad_h() && !conv_param.has_input_pad_w()))
          << "pad is pad OR pad_h and pad_w are required.";
          
      CHECK((!conv_param.has_output_pad() && conv_param.has_output_pad_h()
          && conv_param.has_output_pad_w())
          || (!conv_param.has_output_pad_h() && !conv_param.has_output_pad_w()))
          << "pad is pad OR pad_h and pad_w are required.";
          
      CHECK((!conv_param.has_input_stride() && conv_param.has_input_stride_h()
          && conv_param.has_input_stride_w())
          || (!conv_param.has_input_stride_h() && !conv_param.has_input_stride_w()))
          << "Stride is stride OR stride_h and stride_w are required.";
      CHECK((!conv_param.has_output_stride() && conv_param.has_output_stride_h()
          && conv_param.has_output_stride_w())
          || (!conv_param.has_output_stride_h() && !conv_param.has_output_stride_w()))
          << "Stride is stride OR stride_h and stride_w are required.";  
          
      if (conv_param.has_input_patch_size()) {
        input_patch_h_ = input_patch_w_ = conv_param.input_patch_size();
      } else {
        input_patch_h_ = conv_param.input_patch_h();
        input_patch_w_ = conv_param.input_patch_w();
      }
      CHECK_GT(input_patch_h_, 0) << "Filter dimensions cannot be zero.";
      CHECK_GT(input_patch_w_, 0) << "Filter dimensions cannot be zero.";
      
      if (conv_param.has_output_patch_size()) {
        output_patch_h_ = output_patch_w_ = conv_param.output_patch_size();
      } else {
        output_patch_h_ = conv_param.output_patch_h();
        output_patch_w_ = conv_param.output_patch_w();
      }
      CHECK_GT(output_patch_h_, 0) << "Filter dimensions cannot be zero.";
      CHECK_GT(output_patch_w_, 0) << "Filter dimensions cannot be zero.";

      if (!conv_param.has_input_pad_h()) {
        input_pad_h_ = input_pad_w_ = conv_param.input_pad();
      } else {
        input_pad_h_ = conv_param.input_pad_h();
        input_pad_w_ = conv_param.input_pad_w();
      }
      
      if (!conv_param.has_output_pad_h()) {
        output_pad_h_ = output_pad_w_ = conv_param.output_pad();
      } else {
        output_pad_h_ = conv_param.output_pad_h();
        output_pad_w_ = conv_param.output_pad_w();
      }
      
      if (!conv_param.has_input_stride_h()) {
        input_stride_h_ = input_stride_w_ = conv_param.input_stride();
      } else {
        input_stride_h_ = conv_param.input_stride_h();
        input_stride_w_ = conv_param.input_stride_w();
      }
      
      if (!conv_param.has_output_stride_h()) {
        output_stride_h_ = output_stride_w_ = conv_param.output_stride();
      } else {
        output_stride_h_ = conv_param.output_stride_h();
        output_stride_w_ = conv_param.output_stride_w();
      }

      // Special case: im2col is the identity for 1x1 convolution with stride 1
      // and no padding, so flag for skipping the buffer and transformation.
      input_is_1x1_ = input_patch_w_ == 1 && input_patch_h_ == 1
          && input_stride_h_ == 1 && input_stride_w_ == 1 && input_pad_h_ == 0 && input_pad_w_ == 0;
          
      output_is_1x1_ = output_patch_w_ == 1 && output_patch_h_ == 1
          && output_stride_h_ == 1 && output_stride_w_ == 1 && output_pad_h_ == 0 && output_pad_w_ == 0;
  
      // Configure output channels and groups.
      input_channels_ = bottom[0]->channels();
      num_output_ = this->layer_param_.regionconvolution_param().num_output();
      CHECK_GT(num_output_, 0);
      group_ = this->layer_param_.regionconvolution_param().group();
      CHECK_EQ(input_channels_ % group_, 0);
      CHECK_EQ(num_output_ % group_, 0)
          << "Number of output should be multiples of group.";
          
      weight_w_ = input_patch_w_ * input_patch_h_ * input_channels_/ group_;
      weight_h_ = output_patch_w_ * output_patch_h_ * num_output_;
      
      // Handle the parameters: weights and biases.
      // - blobs_[0] holds the filter weights
      // - blobs_[1] holds the biases (optional)
      bias_term_ = this->layer_param_.regionconvolution_param().bias_term();
      if (this->blobs_.size() > 0) {
        LOG(INFO) << "Skipping parameter initialization";
      } else {
        if (bias_term_) {
          this->blobs_.resize(2);
        } else {
          this->blobs_.resize(1);
        }
        // Initialize and fill the weights:
        this->blobs_[0].reset(new Blob<Dtype>(
            1,1, weight_h_, weight_w_));
            
        shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
            this->layer_param_.regionconvolution_param().weight_filler()));
        weight_filler->Fill(this->blobs_[0].get());
        
        
        // If necessary, initialize and fill the biases.
        if (bias_term_) {
          vector<int> bias_shape(1, num_output_);
          this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
          shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
              this->layer_param_.regionconvolution_param().bias_filler()));
          bias_filler->Fill(this->blobs_[1].get());
        }
      }
      // Propagate gradients to the parameters (as directed by backward pass).
      this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void RegionconvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      
      CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
      num_ = bottom[0]->num();
      input_height_ = bottom[0]->height();
      input_width_ = bottom[0]->width();
      CHECK_EQ(bottom[0]->channels(), input_channels_) << "Input size incompatible with"
        " convolution kernel.";
      // TODO: generalize to handle inputs of different shapes.
      for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
        CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
        CHECK_EQ(input_channels_, bottom[bottom_id]->channels())
            << "Inputs must have same channels.";
        CHECK_EQ(input_height_, bottom[bottom_id]->height())
            << "Inputs must have same height.";
        CHECK_EQ(input_width_, bottom[bottom_id]->width())
            << "Inputs must have same width.";
      }
      
      input_col_height_ = (this->input_height_ + 2 * this->input_pad_h_ - this->input_patch_h_)/ this->input_stride_h_ + 1;
      input_col_width_ = (this->input_width_ + 2 * this->input_pad_w_ - this->input_patch_w_)/ this->input_stride_w_ + 1;
      this->height_out_ = this->output_stride_h_ * (input_col_height_ - 1) + this->output_patch_h_ - 2 * this->output_pad_h_;
      this->width_out_ = this->output_stride_w_ * (input_col_width_ - 1) + this->output_patch_w_ - 2 * this->output_pad_w_;
      
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
      }
     
      weight_offset_ = (output_patch_h_ * output_patch_w_ * num_output_)* (input_patch_h_ * input_patch_w_ * input_channels_) / group_ / group_;
      input_col_offset_ =  (input_patch_h_ * input_patch_w_ * input_channels_)* (input_col_height_ * input_col_width_) / group_;
      output_col_offset_ =  (output_patch_h_ * output_patch_w_ * num_output_)* (input_col_height_ * input_col_width_) / group_;
      
      input_col_buffer_.Reshape(1, 1, 1, (input_patch_h_ * input_patch_w_ * input_channels_)* (input_col_height_ * input_col_width_));
      output_col_buffer_.Reshape(1, 1, 1, (output_patch_h_ * output_patch_w_ * num_output_)* (input_col_height_ * input_col_width_));
      // Set up the all ones "bias multiplier" for adding biases by BLAS
      if (bias_term_) {
        vector<int> bias_multiplier_shape(1, height_out_ * width_out_);
        bias_multiplier_.Reshape(bias_multiplier_shape);
        caffe_set(bias_multiplier_.count(), Dtype(1),
            bias_multiplier_.mutable_cpu_data());
      }
      
      M_ = num_output_ * output_pad_h_ * output_pad_w_ / group_; 
      N_ = input_col_height_ * input_col_width_;
      K_ = input_channels_ * input_patch_h_ * input_patch_w_ / group_;
}

template <typename Dtype>
void RegionconvolutionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
    for (int i = 0; i < bottom.size(); ++i) {
        
        const Dtype* bottom_data = bottom[i]->cpu_data();
        Dtype* top_data = top[i]->mutable_cpu_data();
        Dtype* input_col_data = input_col_buffer_.mutable_cpu_data();
        Dtype* output_col_data = output_col_buffer_.mutable_cpu_data();
        const Dtype* weight = this->blobs_[0]->cpu_data();
        
        for (int n = 0; n < num_; ++n) {	  
            im2col_cpu(bottom_data + bottom[i]->offset(n), input_channels_, input_height_,
                      input_width_, input_patch_h_, input_patch_w_, input_pad_h_, input_pad_w_, input_stride_h_, input_stride_w_,
                      input_col_data);    
                      
            for (int g = 0; g < group_; ++g) {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
                  (Dtype)1., weight + weight_offset_ * g , input_col_data + input_col_offset_ * g,
                  (Dtype)0., output_col_data + output_col_offset_ * g);
            }
            
            col2im_cpu(output_col_data, num_output_, height_out_, width_out_,
                    output_patch_h_, output_patch_w_, output_pad_h_, output_pad_w_,
                    output_stride_h_, output_stride_w_, top_data + top[i]->offset(n));
   
            if (bias_term_) {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                    height_out_ * width_out_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
                    bias_multiplier_.cpu_data(),
                    (Dtype)1., top_data + top[i]->offset(n));
            }  
        }
        
      }
    
}

template <typename Dtype>
void RegionconvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      
      const Dtype* weight = NULL;
      Dtype* weight_diff = NULL;
      if (this->param_propagate_down_[0]) {
        weight = this->blobs_[0]->cpu_data();
        weight_diff = this->blobs_[0]->mutable_cpu_diff();
        caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
      }
      Dtype* bias_diff = NULL;
      if (bias_term_ && this->param_propagate_down_[1]) {
        bias_diff = this->blobs_[1]->mutable_cpu_diff();
        caffe_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
      }

      for (int i = 0; i < top.size(); ++i) {
      
        const Dtype* top_diff = NULL;
        // Bias gradient, if necessary.
        if (bias_term_ && this->param_propagate_down_[1]) {
          top_diff = top[i]->cpu_diff();
          for (int n = 0; n < num_; ++n) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                     1,height_out_ * width_out_, (Dtype)1.,top_diff + top[i]->offset(n) ,
                    bias_multiplier_.cpu_data(),
                    (Dtype)1., bias_diff);
          }
        }
        if (this->param_propagate_down_[0] || propagate_down[i]) {
          if (!top_diff) {
            top_diff = top[i]->cpu_diff();
          }
          
          Dtype* input_col_data = input_col_buffer_.mutable_cpu_data();
          Dtype* input_col_diff = input_col_buffer_.mutable_cpu_diff();
          Dtype* output_col_data = output_col_buffer_.mutable_cpu_data();
          Dtype* output_col_diff = output_col_buffer_.mutable_cpu_diff();

          const Dtype* bottom_data = bottom[i]->cpu_data();
          Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
          
          
          for (int n = 0; n < num_; ++n) {
            
            im2col_cpu(bottom_data + bottom[i]->offset(n), input_channels_, input_height_,
                      input_width_, input_patch_h_, input_patch_w_, input_pad_h_, input_pad_w_, input_stride_h_, input_stride_w_,
                      input_col_data);    
            im2col_cpu(top_diff + top[i]->offset(n), num_output_, height_out_,
                       width_out_, output_patch_h_, output_patch_w_, output_pad_h_, output_pad_w_, output_stride_h_, output_stride_w_,
                      output_col_diff);  
                                         
            // gradient w.r.t. weight. Note that we will accumulate diffs.
            if (this->param_propagate_down_[0]) {
              for (int g = 0; g < group_; ++g) {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                    (Dtype)1., output_col_diff + output_col_offset_ * g,
                    input_col_data + input_col_offset_ * g, (Dtype)1.,
                    weight_diff + weight_offset_ * g);
              }
            }
            // gradient w.r.t. bottom data, if necessary.
            if (propagate_down[i]) {
              if (weight == NULL) {
                weight = this->blobs_[0]->cpu_data();
              }
              for (int g = 0; g < group_; ++g) {
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                    (Dtype)1., weight + weight_offset_ * g,
                    output_col_diff + output_col_offset_ * g,
                    (Dtype)0., input_col_diff + input_col_offset_ * g);
              }
              // col2im back to the data                
              col2im_cpu(input_col_diff, input_channels_, input_height_, input_width_,
                    input_patch_h_, input_patch_w_, input_pad_h_, input_pad_w_,
                    input_stride_h_, input_stride_w_, bottom_diff + bottom[i]->offset(n));
            }
          }
        }
      }
}

#ifdef CPU_ONLY
STUB_GPU(RegionconvolutionLayer);
#endif

INSTANTIATE_CLASS(RegionconvolutionLayer);
REGISTER_LAYER_CLASS(Regionconvolution);

}  // namespace caffe
