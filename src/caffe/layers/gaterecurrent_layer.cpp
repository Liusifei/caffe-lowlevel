#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void GateRecurrentLayer<Dtype>::active_Forward_cpu(const int n, Dtype * data)
{
	switch (this->layer_param_.gaterecurrent_param().active()) 
	{
		case SpatialRecurrentParameter_Active_LINEAR:
			//do nothing
			break;
		case SpatialRecurrentParameter_Active_SIGMOID:
            caffe_cpu_sigmoid_forward(n,data,data);
			break;
		case SpatialRecurrentParameter_Active_RELU:
            caffe_cpu_relu_forward(n,data,data);
			break;
        case SpatialRecurrentParameter_Active_TANH:
            caffe_cpu_tanh_forward(n,data,data);
			break;
		default:
			LOG(FATAL) << "Unknown active method.";
	}
}

template <typename Dtype>
void GateRecurrentLayer<Dtype>::active_Backward_cpu(const int n, const Dtype * data, Dtype * diff)
{
	switch (this->layer_param_.gaterecurrent_param().active()) 
	{
		case SpatialRecurrentParameter_Active_LINEAR:
			//do nothing
			break;
		case SpatialRecurrentParameter_Active_SIGMOID:
            caffe_cpu_sigmoid_backward(n,data,diff,diff);
			break;
		case SpatialRecurrentParameter_Active_RELU:
            caffe_cpu_relu_backward(n,data,diff,diff);
			break;
        case SpatialRecurrentParameter_Active_TANH:
            caffe_cpu_tanh_backward(n,data,diff,diff);
			break;
		default:
			LOG(FATAL) << "Unknown active method.";
	}
}

template <typename Dtype>
void GateRecurrentLayer<Dtype>::disorder_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse, int channels)
{
 
	const int dims []= {num_,channels,height_,width_};
	const int horizontal_disorder []= {3,0,2,1};
    const int vertical_disorder []= {2,0,3,1};
	const int dimsize = 4;
	if(horizontal_ && ! reverse_)
	{// left --> right
		caffe_cpu_permute(datain,dataout,dims,horizontal_disorder,dimsize,-1);
	}
	else if(horizontal_ &&  reverse_)
	{// right --> left
		caffe_cpu_permute(datain,dataout,dims,horizontal_disorder,dimsize,0);
	}
	else if( !horizontal_ && !reverse_)
	{// top --> bottom
		caffe_cpu_permute(datain,dataout,dims,vertical_disorder,dimsize,-1);
	}
	else
	{// bottom --> top
		caffe_cpu_permute(datain,dataout,dims,vertical_disorder,dimsize,0);
	}
	return;
}
template <typename Dtype>
void GateRecurrentLayer<Dtype>::reorder_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse,int channels)
{
	
	
	const int horizontal_recoverdims []= {width_,num_,height_,channels};
    const int vertical_recoverdims []= {height_,num_,width_,channels};
	const int horizontal_recoverorder []= {1,3,2,0};
    const int vertical_recoverorder []= {1,3,0,2};
	const int dimsize = 4;
	
	if(horizontal_ && ! reverse_)
	{// left --> right
		caffe_cpu_permute(datain,dataout,horizontal_recoverdims,horizontal_recoverorder,dimsize,-1);
	}
	else if(horizontal_ &&  reverse_)
	{// right --> left
		caffe_cpu_permute(datain,dataout,horizontal_recoverdims,horizontal_recoverorder,dimsize,3);
	}
	else if( !horizontal_ && !reverse_)
	{// top --> bottom
		caffe_cpu_permute(datain,dataout,vertical_recoverdims,vertical_recoverorder,dimsize,-1);
	}
	else
	{// bottom --> top
		caffe_cpu_permute(datain,dataout,vertical_recoverdims,vertical_recoverorder,dimsize,2);
	}
	return;
}

template <typename Dtype>
void GateRecurrentLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK((bottom.size()==1||bottom.size()==2))<<"bottom size can only be 1 or 2";
    CHECK(top.size()==1)<<"top size must equal to 1";
    
    this->gate_control_=false;
    if(bottom.size()==2)
        this->gate_control_=true;
 
    channels_ = bottom[0]->channels();
    height_out_ = bottom[0]->height();
    width_out_ = bottom[0]->width();

    num_output_ = this->layer_param_.gaterecurrent_param().num_output();
     
    horizontal_ = this->layer_param_.gaterecurrent_param().horizontal();
    reverse_ = this->layer_param_.gaterecurrent_param().reverse();
    use_bias_ = this->layer_param_.gaterecurrent_param().use_bias();
    use_wx_ = this->layer_param_.gaterecurrent_param().use_wx();
    use_wh_ = this->layer_param_.gaterecurrent_param().use_wh();
    restrict_g_ = this->layer_param_.gaterecurrent_param().restrict_g();
    
    if(!use_wx_)
    {
        CHECK(channels_ == num_output_)<<"if you do not use Wx, then channels need equal to num_output, ask liangji for details";
    }
    
    
    if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
    } else {
      this->blobs_.resize(3);
    }

    //Wx
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, channels_, num_output_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.gaterecurrent_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    //Wh
    this->blobs_[1].reset(new Blob<Dtype>(1, 1, num_output_, num_output_));
    weight_filler->Fill(this->blobs_[1].get());

   
      this->blobs_[2].reset(new Blob<Dtype>(1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.gaterecurrent_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    

    this->param_propagate_down_.resize(this->blobs_.size(), true);
    
    this->bound_diff_threshold_ = this->layer_param_.gaterecurrent_param().bound_diff();
    this->restrict_w_ = this->layer_param_.gaterecurrent_param().restrict_w();    
}

template <typename Dtype>
void GateRecurrentLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    num_ = bottom[0]->num();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    CHECK_EQ(bottom[0]->channels(), channels_) << "Input channels incompatible in spatial recurrent reshape";
    CHECK_EQ(height_, height_out_) << "Input height incompatible in spatial recurrent reshape";
    CHECK_EQ(width_, width_out_) << "Input width incompatible in spatial recurrent reshape";


    if(this->horizontal_)
    {
        T_ = width_;
        col_length_ = height_;
    }
    else
    {
        T_ = height_;
        col_length_ = width_; 
    }
    //col_count_ = col_length_ * num_ * channels_;
    
    if(bottom.size()==2)
    {
        CHECK_EQ(num_, bottom[1]->num());
        CHECK_EQ(height_, bottom[1]->height());
        CHECK_EQ(width_, bottom[1]->width());
        CHECK_EQ(num_output_, bottom[1]->channels())<<"gate channels must equal to num_output";
    }

    // Shape the tops.
    for (int top_id = 0; top_id < top.size(); ++top_id) {
        top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
    }

    x_disorder_buffer_.Reshape(num_,channels_,height_,width_);
    h_disorder_buffer_.Reshape(num_,num_output_,height_,width_);
    
    if(this->gate_control_)
    {
        this->gate_disorder_buffer_.Reshape(num_,num_output_,height_,width_);
        this->L_data_buffer_.Reshape(num_,num_output_,height_,width_);
    }

    
    bias_multiplier_.Reshape(1, 1, 1, num_ * col_length_);
    caffe_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_cpu_data());
    
}

template <typename Dtype>
void GateRecurrentLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
    //restrict W_h
    if(restrict_w_ > 0)
    {
        caffe_bound(this->blobs_[1]->count(), this->blobs_[1]->cpu_data(), Dtype(-restrict_w_), Dtype(restrict_w_), this->blobs_[1]->mutable_cpu_data());
    }
    
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* x_data = x_disorder_buffer_.mutable_cpu_data();
    
    const Dtype* W_x = this->blobs_[0]->cpu_data();
    const Dtype* W_h = this->blobs_[1]->cpu_data();
    const Dtype* bias_data = this->blobs_[2]->cpu_data();
    
    
    //get x data
    disorder_inputdata((const Dtype *)bottom_data,x_data,horizontal_,reverse_,channels_);
   
    
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
        disorder_inputdata((const Dtype *)bottom[1]->cpu_data(),this->gate_disorder_buffer_.mutable_cpu_data(),horizontal_,reverse_,num_output_);
        
        Dtype* Gate_data = this->gate_disorder_buffer_.mutable_cpu_data();
        Dtype* H_data = this->h_disorder_buffer_.mutable_cpu_data();
        Dtype* L_data = this->L_data_buffer_.mutable_cpu_data();
        
        if(!use_bias_)
            caffe_set(L_data_buffer_.count(), Dtype(0), L_data);
        
        if(restrict_g_ < 1)
            caffe_scal(gate_disorder_buffer_.count(), restrict_g_, Gate_data);
            
        for(int t=0; t < T_; t++)
        {// finish gate rnn in this loop
            Dtype* G_t = Gate_data + t * G_count;
            Dtype* L_t = L_data + t * L_count;
            Dtype* X_t = x_data + t * X_count;
            Dtype* H_t = H_data + t * H_count;
        
            //L(t)=b
            if(use_bias_)
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
                    (Dtype)1.,bias_multiplier_.cpu_data(), bias_data, 
                    (Dtype)0., L_t );
                    
            //L(t) -= X(t) * W_x
            if(use_wx_)
            {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_x_,
                        Dtype(-1),X_t, W_x, 
                        (Dtype)1., L_t );
            }
            else
            {
                caffe_sub(L_count,L_t,X_t,L_t);
            }
                    
            //L(t) += H(t-1) * W_h if t > 0
            if(t > 0)
            {   
                if(use_wh_)
                {
                    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_h_,
                        (Dtype)1., H_data+ (t-1)* H_count, W_h,
                        (Dtype)1., L_t );
                }
                else
                {
                    caffe_add(L_count,L_t,H_data+ (t-1)* H_count,L_t);
                }
            }
            
            // save G(t).*L(t) in H(t)
            caffe_mul<Dtype>(H_count, G_t, L_t, H_t);
            
            //H(t) += a * X(t)*W_x
            if(use_wx_)
            {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_x_,
                        Dtype(restrict_g_),X_t, W_x, 
                        (Dtype)1., H_t );
            }
            else
            {
                caffe_cpu_axpby<Dtype>(H_count, restrict_g_, X_t,1, H_t);
            }
            
            //active H(t)
            active_Forward_cpu(H_count, H_t );
        }
    }
    else
    {
        CHECK(false)<<"not implement none gate rnn";
    }
    //then recover order to top data
    reorder_outputdata((const Dtype *)h_disorder_buffer_.cpu_data(),top_data,horizontal_,reverse_,num_output_);

}

template <typename Dtype>
void GateRecurrentLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    if (this->param_propagate_down_[0]) {
	caffe_set(this->blobs_[0]->count(),Dtype(0),this->blobs_[0]->mutable_cpu_diff());
    	caffe_set(this->blobs_[1]->count(),Dtype(0),this->blobs_[1]->mutable_cpu_diff());
    	caffe_set(this->blobs_[2]->count(),Dtype(0),this->blobs_[2]->mutable_cpu_diff());
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
    W_x = this->blobs_[0]->cpu_data();
    W_x_diff = this->blobs_[0]->mutable_cpu_diff();
    W_h = this->blobs_[1]->cpu_data();
    W_h_diff = this->blobs_[1]->mutable_cpu_diff();
      
    Dtype* bias_diff = NULL;
    bias_diff = this->blobs_[2]->mutable_cpu_diff();

    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype * H_data = h_disorder_buffer_.cpu_data();
    const Dtype * X_data = x_disorder_buffer_.cpu_data();
    const Dtype * L_data = L_data_buffer_.cpu_data();
    
    Dtype* H_diff = h_disorder_buffer_.mutable_cpu_diff();
    Dtype* X_diff = x_disorder_buffer_.mutable_cpu_diff();
    Dtype* L_diff = L_data_buffer_.mutable_cpu_diff();
  
    //H_diff = top_diff
    disorder_inputdata((const Dtype *)top_diff,H_diff,horizontal_,reverse_,num_output_);
    
    if(gate_control_)
    {
        const Dtype* G_data = this->gate_disorder_buffer_.cpu_data();
        Dtype* G_diff = this->gate_disorder_buffer_.mutable_cpu_diff();
        
        
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
                    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_h_,N_, 
                            (Dtype)1., L_diff + (t+1)*L_count,  W_h,
                            (Dtype)1., H_t_diff);
                }
                else
                {
                    caffe_add(H_count, H_t_diff, L_diff + (t+1)*L_count, H_t_diff);
                }
            }
            
            //active_backward H(t)_diff
            active_Backward_cpu(H_count,H_t,H_t_diff);
            
            //G(t)_diff = H(t)_diff .* L(t)
            caffe_mul<Dtype>(G_count, H_t_diff, L_t, G_t_diff);
            
            //L(t)_diff = H(t)_diff .* G(t)
            caffe_mul<Dtype>(L_count, H_t_diff, G_t, L_t_diff);
            
            //X(t)_diff = a * H(t)_diff * W_x'
            if(use_wx_)
            {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_x_, N_,
                        (Dtype)restrict_g_, H_t_diff,  W_x,
                        (Dtype)0., X_t_diff);
            }
            else
            {
                caffe_cpu_axpby<Dtype>(X_count, restrict_g_, H_t_diff,0, X_t_diff);
                //caffe_copy(X_count,H_t_diff,X_t_diff);
            }
            
            //X(t)_diff -= L(t)_diff * W_x'
            if(use_wx_)
            {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_x_,N_, 
                        (Dtype)-1., L_t_diff,  W_x,
                        (Dtype)1., X_t_diff);
            }
            else
            {
                caffe_sub(X_count,X_t_diff,L_t_diff,X_t_diff);
            }
            
            //W_h_diff += H(t-1)' * L(t)_diff if t > 0}
            if(t>0 && use_wh_)
            {
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_h_, N_, M_,
                        (Dtype)1., H_data + (t-1)* H_count,  L_t_diff,
                        (Dtype)1., W_h_diff);
            }
            
            if(use_wx_)
            {
                //W_x_diff += a * X(t)' * H(t)_diff
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_x_, N_, M_,
                            (Dtype)restrict_g_, X_t,  H_t_diff,
                            (Dtype)1., W_x_diff);
                
                //W_x_diff -= X(t)' * L(t)_diff
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_x_, N_, M_,
                            (Dtype)-1., X_t,  L_t_diff,
                            (Dtype)1., W_x_diff);
            }
            
            //b_diff += L(t)_diff 
            if(use_bias_)
            {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, M_,
                       (Dtype)1., bias_multiplier_.cpu_data(),L_t_diff,
                       (Dtype)1., bias_diff);
            }
            
        
        }
        if(propagate_down[0])
        {
            reorder_outputdata((const Dtype *)X_diff,bottom[0]->mutable_cpu_diff(),horizontal_,reverse_,channels_);
        }
        if(propagate_down[1])
        {
            if(restrict_g_ < 1)
                caffe_scal(gate_disorder_buffer_.count(), restrict_g_, G_diff);
            reorder_outputdata((const Dtype *)G_diff,bottom[1]->mutable_cpu_diff(),horizontal_,reverse_,num_output_);
        }
    }   
    else
    {
        CHECK(false)<<"not implement";
    }
}

#ifdef CPU_ONLY
STUB_GPU(GateRecurrentLayer);
#endif

INSTANTIATE_CLASS(GateRecurrentLayer);
REGISTER_LAYER_CLASS(GateRecurrent);

}  // namespace caffe
