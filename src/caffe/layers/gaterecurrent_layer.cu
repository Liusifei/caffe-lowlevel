#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void GateRecurrentLayer<Dtype>::active_Forward_gpu(const int n, Dtype * data)
{
	switch (this->layer_param_.spatialrecurrent_param().active()) 
	{
		case SpatialRecurrentParameter_Active_LINEAR:
			//do nothing
			break;
		case SpatialRecurrentParameter_Active_SIGMOID:
            caffe_gpu_sigmoid_forward(n,data,data);
			break;
		case SpatialRecurrentParameter_Active_RELU:
            caffe_gpu_relu_forward(n,data,data);
			break;
        case SpatialRecurrentParameter_Active_TANH:
            caffe_gpu_tanh_forward(n,data,data);
			break;
		default:
			LOG(FATAL) << "Unknown active method.";
	}
}

template <typename Dtype>
void GateRecurrentLayer<Dtype>::active_Backward_gpu(const int n, const Dtype * data, Dtype * diff)
{
	switch (this->layer_param_.spatialrecurrent_param().active()) 
	{
		case SpatialRecurrentParameter_Active_LINEAR:
			//do nothing
			break;
		case SpatialRecurrentParameter_Active_SIGMOID:
            caffe_gpu_sigmoid_backward(n,data,diff,diff);
			break;
		case SpatialRecurrentParameter_Active_RELU:
            caffe_gpu_relu_backward(n,data,diff,diff);
			break;
        case SpatialRecurrentParameter_Active_TANH:
            caffe_gpu_tanh_backward(n,data,diff,diff);
			break;
		default:
			LOG(FATAL) << "Unknown active method.";
	}
}

template <typename Dtype>
void GateRecurrentLayer<Dtype>::disorder_gpu_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse,int channels)
{
    const int dims []= {num_,channels,height_,width_};
	const int horizontal_disorder []= {3,0,2,1};
    const int vertical_disorder []= {2,0,3,1};
	const int dimsize = 4;
	if(horizontal_ && ! reverse_)
	{// left --> right
		caffe_gpu_permute(datain,dataout,dims,horizontal_disorder,dimsize,-1);
	}
	else if(horizontal_ &&  reverse_)
	{// right --> left
		caffe_gpu_permute(datain,dataout,dims,horizontal_disorder,dimsize,0);
	}
	else if( !horizontal_ && !reverse_)
	{// top --> bottom
		caffe_gpu_permute(datain,dataout,dims,vertical_disorder,dimsize,-1);
	}
	else
	{// bottom --> top
		caffe_gpu_permute(datain,dataout,dims,vertical_disorder,dimsize,0);
	}
	return;
}

template <typename Dtype>
void GateRecurrentLayer<Dtype>::reorder_gpu_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse,int channels)
{
    
	const int horizontal_recoverdims []= {width_,num_,height_,channels};
    const int vertical_recoverdims []= {height_,num_,width_,channels};
	const int horizontal_recoverorder []= {1,3,2,0};
    const int vertical_recoverorder []= {1,3,0,2};
	const int dimsize = 4;
	
	if(horizontal_ && ! reverse_)
	{// left --> right
		caffe_gpu_permute(datain,dataout,horizontal_recoverdims,horizontal_recoverorder,dimsize,-1);
	}
	else if(horizontal_ &&  reverse_)
	{// right --> left
		caffe_gpu_permute(datain,dataout,horizontal_recoverdims,horizontal_recoverorder,dimsize,3);
	}
	else if( !horizontal_ && !reverse_)
	{// top --> bottom
		caffe_gpu_permute(datain,dataout,vertical_recoverdims,vertical_recoverorder,dimsize,-1);
	}
	else
	{// bottom --> top
		caffe_gpu_permute(datain,dataout,vertical_recoverdims,vertical_recoverorder,dimsize,2);
	}
	return;
}

template <typename Dtype>
void GateRecurrentLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
    //restrict W_h
    if(restrict_w_ > 0)
    {
        caffe_gpu_bound(this->blobs_[1]->count(), this->blobs_[1]->gpu_data(), Dtype(-restrict_w_), Dtype(restrict_w_), this->blobs_[1]->mutable_gpu_data());
    }
    
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* x_data = x_disorder_buffer_.mutable_gpu_data();
    
    const Dtype* W_x = this->blobs_[0]->gpu_data();
    const Dtype* W_h = this->blobs_[1]->gpu_data();
    const Dtype* bias_data = this->blobs_[2]->gpu_data();
    
    
    //get x data
    disorder_gpu_inputdata((const Dtype *)bottom_data,x_data,horizontal_,reverse_,channels_);
   
    
    M_ = num_ * col_length_;
    N_ = num_output_;
    K_x_ = channels_;
    K_h_ = num_output_;
    
    const int X_count = M_*K_x_;
    const int G_count = M_* N_;
    const int L_count = M_* N_;
    const int H_count = M_* N_;
    
    if(this->gate_control_)
    {    
        //get gate data
        disorder_gpu_inputdata((const Dtype *)bottom[1]->gpu_data(),this->gate_disorder_buffer_.mutable_gpu_data(),horizontal_,reverse_,num_output_);
        
        Dtype* Gate_data = this->gate_disorder_buffer_.mutable_gpu_data();
        Dtype* H_data = this->h_disorder_buffer_.mutable_gpu_data();
        Dtype* L_data = this->L_data_buffer_.mutable_gpu_data();
        
        if(!use_bias_)
            caffe_gpu_set(L_data_buffer_.count(), Dtype(0), L_data);
        
        if(restrict_g_ < 1)
            caffe_gpu_scal(gate_disorder_buffer_.count(), restrict_g_, Gate_data);
            
        for(int t=0; t < T_; t++)
        {// finish gate rnn in this loop
            Dtype* G_t = Gate_data + t * G_count;
            Dtype* L_t = L_data + t * L_count;
            Dtype* X_t = x_data + t * X_count;
            Dtype* H_t = H_data + t * H_count;
        
            //L(t)=b
            if(use_bias_)
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
                    (Dtype)1.,bias_multiplier_.gpu_data(), bias_data, 
                    (Dtype)0., L_t );
                    
            //L(t) -= X(t) * W_x
            if(use_wx_)
            {
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_x_,
                        Dtype(-1),X_t, W_x, 
                        (Dtype)1., L_t );
            }
            else
            {
                caffe_gpu_sub(L_count,L_t,X_t,L_t);
            }
                    
            //L(t) += H(t-1) * W_h if t > 0
            if(t > 0)
            {   
                if(use_wh_)
                {
                    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_h_,
                        (Dtype)1., H_data+ (t-1)* H_count, W_h,
                        (Dtype)1., L_t );
                }
                else
                {
                    caffe_gpu_add(L_count,L_t,H_data+ (t-1)* H_count,L_t);
                }
            }
            
            // save G(t).*L(t) in H(t)
            caffe_gpu_mul<Dtype>(H_count, G_t, L_t, H_t);
            
            //H(t) += a * X(t)*W_x
            if(use_wx_)
            {
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_x_,
                        Dtype(restrict_g_),X_t, W_x, 
                        (Dtype)1., H_t );
            }
            else
            {
                caffe_gpu_axpby<Dtype>(H_count, restrict_g_, X_t,1, H_t);
                //caffe_gpu_add(H_count,H_t,X_t,H_t);
            }
            
            //active H(t)
            active_Forward_gpu(H_count, H_t );
        }
    }
    else
    {
        CHECK(false)<<"not implement none gate rnn";
    }
    //then recover order to top data
    reorder_gpu_outputdata((const Dtype *)h_disorder_buffer_.gpu_data(),top_data,horizontal_,reverse_,num_output_);

}

template <typename Dtype>
void GateRecurrentLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    if (this->param_propagate_down_[0]) {
	caffe_gpu_set(this->blobs_[0]->count(),Dtype(0),this->blobs_[0]->mutable_gpu_diff());
        caffe_gpu_set(this->blobs_[1]->count(),Dtype(0),this->blobs_[1]->mutable_gpu_diff());
        caffe_gpu_set(this->blobs_[2]->count(),Dtype(0),this->blobs_[2]->mutable_gpu_diff());
    }
 
    M_ = num_ * col_length_;
    N_ = num_output_;
    K_x_ = channels_;
    K_h_ = num_output_;
    
    const int X_count = M_*K_x_;
    const int G_count = M_* N_;
    const int L_count = M_* N_;
    const int H_count = M_* N_;
    
    const Dtype* W_x = NULL;
    Dtype* W_x_diff = NULL;
    const Dtype* W_h = NULL;
    Dtype* W_h_diff = NULL;
    W_x = this->blobs_[0]->gpu_data();
    W_x_diff = this->blobs_[0]->mutable_gpu_diff();
    W_h = this->blobs_[1]->gpu_data();
    W_h_diff = this->blobs_[1]->mutable_gpu_diff();
      
    Dtype* bias_diff = NULL;
    bias_diff = this->blobs_[2]->mutable_gpu_diff();

    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype * H_data = h_disorder_buffer_.gpu_data();
    const Dtype * X_data = x_disorder_buffer_.gpu_data();
    const Dtype * L_data = L_data_buffer_.gpu_data();
    
    Dtype* H_diff = h_disorder_buffer_.mutable_gpu_diff();
    Dtype* X_diff = x_disorder_buffer_.mutable_gpu_diff();
    Dtype* L_diff = L_data_buffer_.mutable_gpu_diff();
  
    //H_diff = top_diff
    disorder_gpu_inputdata((const Dtype *)top_diff,H_diff,horizontal_,reverse_,num_output_);
    
    if(gate_control_)
    {
        const Dtype* G_data = this->gate_disorder_buffer_.gpu_data();
        Dtype* G_diff = this->gate_disorder_buffer_.mutable_gpu_diff();
        
        
        for(int t= T_ - 1; t >= 0; t--)
        {//finish right to left gate rnn BP in this loop
            
            const Dtype* H_t = H_data + t*H_count;
            const Dtype* X_t = X_data + t*X_count;
            const Dtype* L_t = L_data + t*L_count;
            const Dtype* G_t = G_data + t*G_count;
            
            
            Dtype* L_t_diff = L_diff + t*L_count;
            Dtype* H_t_diff = H_diff + t*H_count;
            Dtype* G_t_diff = G_diff + t*G_count;
            Dtype* X_t_diff = X_diff + t*X_count;
            
        
            //H(t)_diff += L(t+1)_diff * W_h'  if t < T-1
            if(t < T_-1)
            {
                if(use_wh_)
                {
                    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_h_,N_, 
                            (Dtype)1., L_diff + (t+1)*L_count,  W_h,
                            (Dtype)1., H_t_diff);
                }
                else
                {
                    caffe_gpu_add(H_count, H_t_diff, L_diff + (t+1)*L_count, H_t_diff);
                }
            }
            
            //active_backward H(t)_diff
            active_Backward_gpu(H_count,H_t,H_t_diff);
            
            //G(t)_diff = H(t)_diff .* L(t)
            caffe_gpu_mul<Dtype>(G_count, H_t_diff, L_t, G_t_diff);
            
            //L(t)_diff = H(t)_diff .* G(t)
            caffe_gpu_mul<Dtype>(L_count, H_t_diff, G_t, L_t_diff);
            
            //X(t)_diff = a * H(t)_diff * W_x'
            if(use_wx_)
            {
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_x_, N_,
                        (Dtype)restrict_g_, H_t_diff,  W_x,
                        (Dtype)0., X_t_diff);
            }
            else
            {
                caffe_gpu_axpby<Dtype>(X_count, restrict_g_, H_t_diff,0, X_t_diff);
                //caffe_copy(X_count,H_t_diff,X_t_diff);
            }
            
            //X(t)_diff -= L(t)_diff * W_x'
            if(use_wx_)
            {
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_x_,N_, 
                        (Dtype)-1., L_t_diff,  W_x,
                        (Dtype)1., X_t_diff);
            }
            else
            {
                caffe_gpu_sub(X_count,X_t_diff,L_t_diff,X_t_diff);
            }
            
            //W_h_diff += H(t-1)' * L(t)_diff if t > 0}
            if(t>0 && use_wh_)
            {
                caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_h_, N_, M_,
                        (Dtype)1., H_data + (t-1)* H_count,  L_t_diff,
                        (Dtype)1., W_h_diff);
            }
            
            if(use_wx_)
            {
                //W_x_diff += a * X(t)' * H(t)_diff
                caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_x_, N_, M_,
                            (Dtype)restrict_g_, X_t,  H_t_diff,
                            (Dtype)1., W_x_diff);
                
                //W_x_diff -= X(t)' * L(t)_diff
                caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_x_, N_, M_,
                            (Dtype)-1., X_t,  L_t_diff,
                            (Dtype)1., W_x_diff);
            }
            
            //b_diff += L(t)_diff 
            if(use_bias_)
            {
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, M_,
                       (Dtype)1., bias_multiplier_.gpu_data(),L_t_diff,
                       (Dtype)1., bias_diff);
            }
            
        
        }
        if(propagate_down[0])
        {
            reorder_gpu_outputdata((const Dtype *)X_diff,bottom[0]->mutable_gpu_diff(),horizontal_,reverse_,channels_);
        }
        if(propagate_down[1])
        {
            if(restrict_g_ < 1)
                caffe_gpu_scal(gate_disorder_buffer_.count(), restrict_g_, G_diff);
            reorder_gpu_outputdata((const Dtype *)G_diff,bottom[1]->mutable_gpu_diff(),horizontal_,reverse_,num_output_);
        }
    }   
    else
    {
        CHECK(false)<<"not implement";
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(GateRecurrentLayer);

}  // namespace caffe
