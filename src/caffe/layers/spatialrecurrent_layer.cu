#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SpatialRecurrentLayer<Dtype>::active_Forward_gpu(const int n, Dtype * data)
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
void SpatialRecurrentLayer<Dtype>::active_Backward_gpu(const int n, const Dtype * data, Dtype * diff)
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
void SpatialRecurrentLayer<Dtype>::disorder_gpu_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse)
{
    const int dims []= {num_,channels_,height_,width_};
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
void SpatialRecurrentLayer<Dtype>::reorder_gpu_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse)
{
    
	const int horizontal_recoverdims []= {width_,num_,height_,channels_};
    const int vertical_recoverdims []= {height_,num_,width_,channels_};
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
void SpatialRecurrentLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
    if(restrict_w_ > 0)
    {
        caffe_gpu_bound(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(), Dtype(-restrict_w_), Dtype(restrict_w_), this->blobs_[0]->mutable_gpu_data());
    }
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* col_data = col_buffer_.mutable_gpu_data();
    Dtype* disorder_data = data_disorder_buffer_.mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    
    
    CHECK(bottom[0]->count() == data_disorder_buffer_.count());
    
    disorder_gpu_inputdata((const Dtype *)bottom_data,disorder_data,horizontal_,reverse_);
    // now disorder_data (h) == x, next we should do x+H*W from left to right, col by col
    
    M_ = num_ * col_length_;
    K_ = channels_;
    N_ = channels_;

    
    if(this->gate_control_)
    {    
        //disorder gate data
        disorder_gpu_inputdata((const Dtype *)bottom[1]->gpu_data(),this->gate_disorder_buffer_.mutable_gpu_data(),horizontal_,reverse_);
        
        const Dtype* gate_data = this->gate_disorder_buffer_.gpu_data();
        Dtype* L_data = this->L_data_buffer_.mutable_gpu_data();
        
        //L[0] = 0*w +b
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
              (Dtype)1., bias_multiplier_.gpu_data(), this->blobs_[1]->gpu_data(),
              (Dtype)0., L_data );  
        //col_data=L[0].*gate[0]
        caffe_gpu_mul<Dtype>(col_count_,L_data,gate_data,col_data);
        //h[0] += col_data
        caffe_gpu_add<Dtype>(col_count_,disorder_data,col_data,disorder_data);      
        //active h(0)
        active_Forward_gpu(col_count_, disorder_data );
        
        for(int col=1; col < T_; col++)
        {//finish left to right rnn in this loop
        
            // L[t]=h[t-1]*w
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
              (Dtype)1., disorder_data+ (col-1)* col_count_ , weight,
              (Dtype)0., L_data + col* col_count_);
              
            //L[t] += b
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
              (Dtype)1., bias_multiplier_.gpu_data(), this->blobs_[1]->gpu_data(),
              (Dtype)1., L_data + col* col_count_ );
              
            //col_data = L[t].*gate[t]
            caffe_gpu_mul<Dtype>(col_count_,L_data+ col* col_count_ ,gate_data+ col* col_count_,col_data);
          
            //h[i] += col_data
            caffe_gpu_add<Dtype>(col_count_,col_data,disorder_data+ col* col_count_,disorder_data+ col* col_count_);      
            
            //active h[i]
            active_Forward_gpu(col_count_, disorder_data+ col* col_count_ );
          
        }
    }
    else
    {
        //h[0] += 0*w + b
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
              (Dtype)1., bias_multiplier_.gpu_data(), this->blobs_[1]->gpu_data(),
              (Dtype)1., disorder_data );  
        //active h(0)
        active_Forward_gpu(col_count_, disorder_data );
        
        for(int col=1; col < T_; col++)
        {//finish left to right rnn in this loop
        
            // col_buffer_ = h[i-1]*w
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
              (Dtype)1., disorder_data+ (col-1)* col_count_ , weight,
              (Dtype)0., col_data );
              
            //col_buffer_ += b
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
              (Dtype)1., bias_multiplier_.gpu_data(), this->blobs_[1]->gpu_data(),
              (Dtype)1., col_data );
          
            //h(i) += col_buffer_
            caffe_gpu_axpy<Dtype>(col_count_, (Dtype)1., col_data, disorder_data+ col* col_count_ );
            
            //active h(i)
            active_Forward_gpu(col_count_, disorder_data+ col* col_count_ );
          
        }
    }
    //then recover order to top data
    CHECK(top[0]->count() == data_disorder_buffer_.count());
    reorder_gpu_outputdata((const Dtype *)disorder_data,top_data,horizontal_,reverse_);
}

template <typename Dtype>
void SpatialRecurrentLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* weight = NULL;
    Dtype* weight_diff = NULL;
  
    if (this->param_propagate_down_[0]) {
        weight = this->blobs_[0]->gpu_data();
        weight_diff = this->blobs_[0]->mutable_gpu_diff();
        caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
    }
    Dtype* bias_diff = NULL;
    if (bias_term_ && this->param_propagate_down_[1]) {
        bias_diff = this->blobs_[1]->mutable_gpu_diff();
        caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
    }
  
     M_ = num_ * col_length_;
    K_ = channels_;
    N_ = channels_;
  
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* col_diff = col_buffer_.mutable_gpu_diff();
    Dtype* disorder_diff = data_disorder_buffer_.mutable_gpu_diff();
    const Dtype* disorder_data = data_disorder_buffer_.gpu_data();
    
    //H(i)_diff = top_diff
    CHECK(top[0]->count() == data_disorder_buffer_.count());
    disorder_gpu_inputdata((const Dtype *)top_diff,disorder_diff,horizontal_,reverse_);
    
    if(gate_control_)
    {
        const Dtype* gate_data = this->gate_disorder_buffer_.gpu_data();
        const Dtype* L_data = this->L_data_buffer_.gpu_data();
        Dtype* gate_diff = this->gate_disorder_buffer_.mutable_gpu_diff();
        
        for(int col= T_ - 2; col >= 0; col--)
        { // update H[t]_diff from right to left
        
            //active_backward, H[i+1]_diff = active_backword(H[i+1]_diff)
            active_Backward_gpu(col_count_,disorder_data + (col+1)* col_count_,disorder_diff+ (col+1)* col_count_);
            
            //col_diff = H[t+1]_diff.*gate[t+1]
            caffe_gpu_mul<Dtype>(col_count_,disorder_diff+ (col+1)* col_count_, gate_data+ (col+1)* col_count_,col_diff);
            
            // h(t)_diff + = col_diff*w'
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
              (Dtype)1., col_diff , weight,
              (Dtype)1., disorder_diff+ col* col_count_ );
        }
        
        //active_backward, H[0]_diff = active_backword(H[0]_diff)
        active_Backward_gpu(col_count_,disorder_data,disorder_diff);
        
        if(propagate_down[0])
        {
            // X[t]_diff = H[t]_diff, recover order to bottom diff
            CHECK(bottom[0]->count() == data_disorder_buffer_.count());
            reorder_gpu_outputdata((const Dtype *)disorder_diff,bottom_diff,horizontal_,reverse_);
        }
        if(propagate_down[1])
        {
            // gate_diff = H_diff .* L_data 
            caffe_gpu_mul<Dtype>(bottom[1]->count(),disorder_diff,L_data,gate_diff);
            // recover order to gate diff
            reorder_gpu_outputdata((const Dtype *)gate_diff,bottom[1]->mutable_gpu_diff(),horizontal_,reverse_);
        }
        
        if(this->param_propagate_down_[0] || this->param_propagate_down_[1])
        {
            //L_diff = H_diff .* gate,  now disorder_diff is L_diff
            caffe_gpu_mul<Dtype>(data_disorder_buffer_.count(),disorder_diff,gate_data ,disorder_diff);
        }
        if(this->param_propagate_down_[0])
        {//finish w diff
            for(int col= 0; col < T_ -1; col++)
            { 
                //col_diff = h[t+1]_diff .* gate[t+1], do not use this because we did the caffe_mul before
                //caffe_mul<Dtype>(col_count,disorder_diff+ (col+1)* col_count,gate_data + (col+1)* col_count,col_diff);
                
                // W_diff += h(t)'*L[t+1]_diff
                caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_,
                            (Dtype)1., disorder_data + col* col_count_, disorder_diff+(col+1)* col_count_,
                            (Dtype)1., weight_diff );
            
            }
        }
        
        if(this->param_propagate_down_[1])
        {//finish b diff
            
            for(int col= 0; col < T_ ; col++)
            {
                //col_diff = h(t)_diff .* gate(t),do not use this because we did the caffe_mul before
                //caffe_mul<Dtype>(col_count,disorder_diff+ col* col_count,gate_data + col* col_count,col_diff);
                
                //b+=L[t]_diff
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, M_,
                            (Dtype)1., bias_multiplier_.gpu_data() , disorder_diff+ col* col_count_,
                            (Dtype)1., bias_diff );
            
            }
        }
    }
    else
    {
        for(int col= T_ - 2; col >= 0; col--)
        { // update H(i)_diff from right to left
        
            //active_backward, H(i+1)_diff = active_backword(H(i+1)_diff)
            active_Backward_gpu(col_count_,disorder_data + (col+1)* col_count_,disorder_diff+ (col+1)* col_count_);
        
            // col_diff = h(i+1)*w'
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
              (Dtype)1., disorder_diff+ (col+1)* col_count_ , weight,
              (Dtype)0., col_diff );
        
            // h(i)_diff += col_diff
            caffe_gpu_axpy<Dtype>(col_count_, (Dtype)1., col_diff, disorder_diff+ col* col_count_ );
        
        }
        
        //active_backward, H(0)_diff = active_backword(H(0)_diff)
        active_Backward_gpu(col_count_,disorder_data,disorder_diff);
        
        
        
        if(propagate_down[0])
        {//finish bottom diff
            // recover order to bottom diff
            CHECK(bottom[0]->count() == data_disorder_buffer_.count());
            reorder_gpu_outputdata((const Dtype *)disorder_diff,bottom_diff,horizontal_,reverse_);
        }
        
        
        
        if(this->param_propagate_down_[0])
        {//finish w diff
            for(int col= 0; col < T_ -1; col++)
            { // w += h(i)'*h(i+1)_diff
                caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_,
                            (Dtype)1., disorder_data + col* col_count_, disorder_diff+ (col+1)* col_count_,
                            (Dtype)1., weight_diff );
            
            }
        }
        
        
        if(this->param_propagate_down_[1])
        {//finish b diff
            
            for(int col= 0; col < T_ ; col++)
            {//b+=h(i)_diff
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, M_,
                            (Dtype)1., bias_multiplier_.gpu_data() , disorder_diff+ col* col_count_,
                            (Dtype)1., bias_diff );
            
            }
        }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialRecurrentLayer);

}  // namespace caffe
