#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SpatialRecurrentLayer<Dtype>::active_Forward_cpu(const int n, Dtype * data)
{
	switch (this->layer_param_.spatialrecurrent_param().active()) 
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
void SpatialRecurrentLayer<Dtype>::active_Backward_cpu(const int n, const Dtype * data, Dtype * diff)
{
	switch (this->layer_param_.spatialrecurrent_param().active()) 
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
void SpatialRecurrentLayer<Dtype>::disorder_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse)
{
 
	const int dims []= {num_,channels_,height_,width_};
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
void SpatialRecurrentLayer<Dtype>::reorder_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse)
{
	
	
	const int horizontal_recoverdims []= {width_,num_,height_,channels_};
    const int vertical_recoverdims []= {height_,num_,width_,channels_};
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
void SpatialRecurrentLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK((bottom.size()==1||bottom.size()==2))<<"bottom size can only be 1 or 2";
    CHECK(top.size()==1)<<"top size must equal to 1";
    
    this->gate_control_=false;
    if(bottom.size()==2)
        this->gate_control_=true;
 
    channels_ = bottom[0]->channels();
    height_out_ = bottom[0]->height();
    width_out_ = bottom[0]->width();

    num_output_ = channels_;  // output size == input size, 
     
    horizontal_ = this->layer_param_.spatialrecurrent_param().horizontal();
    reverse_ = this->layer_param_.spatialrecurrent_param().reverse();

    bias_term_ = this->layer_param_.spatialrecurrent_param().bias_term();
    if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
    } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }


    this->blobs_[0].reset(new Blob<Dtype>(
        1, 1, channels_, channels_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.spatialrecurrent_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());


    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, channels_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.spatialrecurrent_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
    }

    this->param_propagate_down_.resize(this->blobs_.size(), true);
    
    this->bound_diff_threshold_ = this->layer_param_.spatialrecurrent_param().bound_diff();
    this->restrict_w_ = this->layer_param_.spatialrecurrent_param().restrict_w();
    
    //print FLPS
    Dtype flps = 0.0; 
    num_ = bottom[0]->num();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
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
    if(this->gate_control_)
    {
        //y*w
        flps = 2* channels_  * channels_ * (T_ -1)*col_length_;
        //+b
        flps += channels_ * height_out_ * width_out_;
        //.*g
        flps += channels_ * height_out_ * width_out_;
        //+x
        flps += channels_ * height_ * width_;
    }
    else
    {
        //y*w
        flps = 2* channels_ * channels_ * (T_ -1)*col_length_;
        //+b
        flps += channels_ * height_out_ * width_out_;
        //+x
        flps += channels_ * height_ * width_;
    }
    
    flps = flps/1000/1000/1000;
    LOG(INFO)<<this->layer_param_.name()<<" type: "<<this->layer_param_.type()<<" FPLS(G):"<<flps;
}

template <typename Dtype>
void SpatialRecurrentLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
    col_count_ = col_length_ * num_ * channels_;

    // TODO: generalize to handle inputs of different shapes.
    for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
    }

    // Shape the tops.
    for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
    }

    data_disorder_buffer_.Reshape(width_,num_,height_,channels_);
    col_buffer_.Reshape(1, 1,1,col_count_);
    
    if(this->gate_control_)
    {
        this->gate_disorder_buffer_.Reshape(width_,num_,height_,channels_);
        this->L_data_buffer_.Reshape(width_,num_,height_,channels_);
    }

    if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, col_count_ / channels_);
    caffe_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_cpu_data());
    }
}

template <typename Dtype>
void SpatialRecurrentLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
    if(restrict_w_ > 0)
    {
        caffe_bound(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(), Dtype(-restrict_w_), Dtype(restrict_w_), this->blobs_[0]->mutable_cpu_data());
    }
    
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* col_data = col_buffer_.mutable_cpu_data();
    Dtype* disorder_data = data_disorder_buffer_.mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    
    
    
    CHECK(bottom[0]->count() == data_disorder_buffer_.count());
    
    disorder_inputdata((const Dtype *)bottom_data,disorder_data,horizontal_,reverse_);
    // now disorder_data (h) == x, next we should do x+H*W from left to right, col by col
    
    M_ = num_ * col_length_;
    K_ = channels_;
    N_ = channels_;

    
    if(this->gate_control_)
    {    
        //disorder gate data
        disorder_inputdata((const Dtype *)bottom[1]->cpu_data(),this->gate_disorder_buffer_.mutable_cpu_data(),horizontal_,reverse_);
        
        const Dtype* gate_data = this->gate_disorder_buffer_.cpu_data();
        Dtype* L_data = this->L_data_buffer_.mutable_cpu_data();
        
        //L[0] = 0*w +b
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
              (Dtype)1., bias_multiplier_.cpu_data(), this->blobs_[1]->cpu_data(),
              (Dtype)0., L_data );  
        //col_data=L[0].*gate[0]
        caffe_mul<Dtype>(col_count_,L_data,gate_data,col_data);
        //h[0] += col_data
        caffe_add<Dtype>(col_count_,disorder_data,col_data,disorder_data);      
        //active h(0)
        active_Forward_cpu(col_count_, disorder_data );
        
        for(int col=1; col < T_; col++)
        {//finish left to right rnn in this loop
        
            // L[t]=h[t-1]*w
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
              (Dtype)1., disorder_data+ (col-1)* col_count_ , weight,
              (Dtype)0., L_data + col* col_count_);
              
            //L[t] += b
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
              (Dtype)1., bias_multiplier_.cpu_data(), this->blobs_[1]->cpu_data(),
              (Dtype)1., L_data + col* col_count_ );
              
            //col_data = L[t].*gate[t]
            caffe_mul<Dtype>(col_count_,L_data+ col* col_count_ ,gate_data+ col* col_count_,col_data);
          
            //h[i] += col_data
            caffe_add<Dtype>(col_count_,col_data,disorder_data+ col* col_count_,disorder_data+ col* col_count_);      
            
            //active h[i]
            active_Forward_cpu(col_count_, disorder_data+ col* col_count_ );
          
        }
    }
    else
    {
        //h[0] += 0*w + b
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
              (Dtype)1., bias_multiplier_.cpu_data(), this->blobs_[1]->cpu_data(),
              (Dtype)1., disorder_data );  
        //active h(0)
        active_Forward_cpu(col_count_, disorder_data );
        
        for(int col=1; col < T_; col++)
        {//finish left to right rnn in this loop
        
            // col_buffer_ = h[i-1]*w
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
              (Dtype)1., disorder_data+ (col-1)* col_count_ , weight,
              (Dtype)0., col_data );
              
            //col_buffer_ += b
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
              (Dtype)1., bias_multiplier_.cpu_data(), this->blobs_[1]->cpu_data(),
              (Dtype)1., col_data );
          
            //h(i) += col_buffer_
            caffe_axpy<Dtype>(col_count_, (Dtype)1., col_data, disorder_data+ col* col_count_ );
            
            //active h(i)
            active_Forward_cpu(col_count_, disorder_data+ col* col_count_ );
          
        }
    }
    //then recover order to top data
    CHECK(top[0]->count() == data_disorder_buffer_.count());
    reorder_outputdata((const Dtype *)disorder_data,top_data,horizontal_,reverse_);
}

template <typename Dtype>
void SpatialRecurrentLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
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
  
     M_ = num_ * col_length_;
    K_ = channels_;
    N_ = channels_;
  
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype* col_diff = col_buffer_.mutable_cpu_diff();
    Dtype* disorder_diff = data_disorder_buffer_.mutable_cpu_diff();
    const Dtype* disorder_data = data_disorder_buffer_.cpu_data();
    
    //H(i)_diff = top_diff
    CHECK(top[0]->count() == data_disorder_buffer_.count());
    disorder_inputdata((const Dtype *)top_diff,disorder_diff,horizontal_,reverse_);
    
    if(gate_control_)
    {
        const Dtype* gate_data = this->gate_disorder_buffer_.cpu_data();
        const Dtype* L_data = this->L_data_buffer_.cpu_data();
        Dtype* gate_diff = this->gate_disorder_buffer_.mutable_cpu_diff();
        
        for(int col= T_ - 2; col >= 0; col--)
        { // update H[t]_diff from right to left
        
            //active_backward, H[i+1]_diff = active_backword(H[i+1]_diff)
            active_Backward_cpu(col_count_,disorder_data + (col+1)* col_count_,disorder_diff+ (col+1)* col_count_);
            
            //col_diff = H[t+1]_diff.*gate[t+1]
            caffe_mul<Dtype>(col_count_,disorder_diff+ (col+1)* col_count_, gate_data+ (col+1)* col_count_,col_diff);
            
            // h(t)_diff + = col_diff*w'
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
              (Dtype)1., col_diff , weight,
              (Dtype)1., disorder_diff+ col* col_count_ );
        }
        
        //active_backward, H[0]_diff = active_backword(H[0]_diff)
        active_Backward_cpu(col_count_,disorder_data,disorder_diff);
        
        if(propagate_down[0])
        {
            // X[t]_diff = H[t]_diff, recover order to bottom diff
            CHECK(bottom[0]->count() == data_disorder_buffer_.count());
            reorder_outputdata((const Dtype *)disorder_diff,bottom_diff,horizontal_,reverse_);
        }
        if(propagate_down[1])
        {
            // gate_diff = H_diff .* L_data 
            caffe_mul<Dtype>(bottom[1]->count(),disorder_diff,L_data,gate_diff);
            // recover order to gate diff
            reorder_outputdata((const Dtype *)gate_diff,bottom[1]->mutable_cpu_diff(),horizontal_,reverse_);
        }
        
        if(this->param_propagate_down_[0] || this->param_propagate_down_[1])
        {
            //L_diff = H_diff .* gate,  now disorder_diff is L_diff
            caffe_mul<Dtype>(data_disorder_buffer_.count(),disorder_diff,gate_data ,disorder_diff);
        }
        if(this->param_propagate_down_[0])
        {//finish w diff
            for(int col= 0; col < T_ -1; col++)
            { 
                //col_diff = h[t+1]_diff .* gate[t+1], do not use this because we did the caffe_mul before
                //caffe_mul<Dtype>(col_count,disorder_diff+ (col+1)* col_count,gate_data + (col+1)* col_count,col_diff);
                
                // W_diff += h(t)'*L[t+1]_diff
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_,
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
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, M_,
                            (Dtype)1., bias_multiplier_.cpu_data() , disorder_diff+ col* col_count_,
                            (Dtype)1., bias_diff );
            
            }
        }
    }
    else
    {
        for(int col= T_ - 2; col >= 0; col--)
        { // update H(i)_diff from right to left
        
            //active_backward, H(i+1)_diff = active_backword(H(i+1)_diff)
            active_Backward_cpu(col_count_,disorder_data + (col+1)* col_count_,disorder_diff+ (col+1)* col_count_);
        
            // col_diff = h(i+1)*w'
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
              (Dtype)1., disorder_diff+ (col+1)* col_count_ , weight,
              (Dtype)0., col_diff );
        
            // h(i)_diff += col_diff
            caffe_axpy<Dtype>(col_count_, (Dtype)1., col_diff, disorder_diff+ col* col_count_ );
        
        }
        
        //active_backward, H(0)_diff = active_backword(H(0)_diff)
        active_Backward_cpu(col_count_,disorder_data,disorder_diff);
        
        
        
        if(propagate_down[0])
        {//finish bottom diff
            // recover order to bottom diff
            CHECK(bottom[0]->count() == data_disorder_buffer_.count());
            reorder_outputdata((const Dtype *)disorder_diff,bottom_diff,horizontal_,reverse_);
        }
        
        
        
        if(this->param_propagate_down_[0])
        {//finish w diff
            for(int col= 0; col < T_ -1; col++)
            { // w += h(i)'*h(i+1)_diff
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_,
                            (Dtype)1., disorder_data + col* col_count_, disorder_diff+ (col+1)* col_count_,
                            (Dtype)1., weight_diff );
            
            }
        }
        
        
        if(this->param_propagate_down_[1])
        {//finish b diff
            
            for(int col= 0; col < T_ ; col++)
            {//b+=h(i)_diff
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, M_,
                            (Dtype)1., bias_multiplier_.cpu_data() , disorder_diff+ col* col_count_,
                            (Dtype)1., bias_diff );
            
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(SpatialRecurrentLayer);
#endif

INSTANTIATE_CLASS(SpatialRecurrentLayer);
REGISTER_LAYER_CLASS(SpatialRecurrent);

}  // namespace caffe
