#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SpatialLstmLayer<Dtype>::disorder_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse)
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
void SpatialLstmLayer<Dtype>::reorder_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse)
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
void SpatialLstmLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
        
        channels_ = bottom[0]->channels();
        height_out_ = bottom[0]->height();
        width_out_ = bottom[0]->width();

        num_output_ = channels_;  // output size == input size, 

        horizontal_ = this->layer_param_.spatiallstm_param().horizontal();
        reverse_ = this->layer_param_.spatiallstm_param().reverse();

        bias_term_ = this->layer_param_.spatiallstm_param().bias_term();
        if (this->blobs_.size() > 0) 
        {
            LOG(INFO) << "Skipping parameter initialization";
        } 
        else {
            if (bias_term_)
            {
                this->blobs_.resize(2);
            } 
            else
            {
                this->blobs_.resize(1);
            }

            //Wxi, Whi, Wxf, Whf, Wxo, Who, Wxg, Whg, 8 w
            this->blobs_[0].reset(new Blob<Dtype>(1, 8, channels_, channels_));
            shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
                this->layer_param_.spatiallstm_param().weight_filler()));
            weight_filler->Fill(this->blobs_[0].get());

            if (bias_term_)
            {
                //bi, bf,bo,bg, 4 b
                this->blobs_[1].reset(new Blob<Dtype>(1, 1, 4, channels_));
                shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                    this->layer_param_.spatiallstm_param().bias_filler()));
                bias_filler->Fill(this->blobs_[1].get());
            }
        }

        this->param_propagate_down_.resize(this->blobs_.size(), true);
        
        identical_multiplier_.Reshape(1, 1, channels_,channels_);
        Dtype* id_data = identical_multiplier_.mutable_cpu_data();
        for (int h=0;h<channels_;h++)
        {
            for(int w=0;w<channels_;w++)
            {
                if(w == h)
                {
                    id_data[h*channels_ + w] = Dtype(1);
                }
                else
                {
                    id_data[h*channels_ + w] = Dtype(0);
                }
            }
        }

 
}

template <typename Dtype>
void SpatialLstmLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      
        num_ = bottom[0]->num();
        height_ = bottom[0]->height();
        width_ = bottom[0]->width();
        CHECK_EQ(bottom[0]->channels(), channels_) << "Input channels incompatible in spatial lstm reshape";
        CHECK_EQ(height_, height_out_) << "Input height incompatible in spatial lstm reshape";
        CHECK_EQ(width_, width_out_) << "Input width incompatible in spatial lstm reshape";

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
        for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) 
        {
            CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
            CHECK_EQ(channels_, bottom[bottom_id]->channels())
            << "Inputs must have same channels.";
            CHECK_EQ(height_, bottom[bottom_id]->height())
            << "Inputs must have same height.";
            CHECK_EQ(width_, bottom[bottom_id]->width())
            << "Inputs must have same width.";
        }

        // Shape the tops.
        for (int top_id = 0; top_id < top.size(); ++top_id) 
        {
            top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
        }

        data_disorder_buffer_.Reshape(width_,num_,height_,channels_);
        C_buffer_.Reshape(width_,num_,height_,channels_);
        H_buffer_.Reshape(width_,num_,height_,channels_);
        Gate_buffer_.Reshape(1,1,width_ * channels_ * 4, col_count_ / channels_ );
        FC_1_buffer_.Reshape(1,1,height_ * num_,channels_);
        col_buffer_.Reshape(1, 1, 1,col_count_);
    
        if (bias_term_) 
        {
            bias_multiplier_.Reshape(1, 1, 1, col_count_ / channels_);
            caffe_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_cpu_data());
        }
}

template <typename Dtype>
void SpatialLstmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    for (int i = 0; i < bottom.size(); ++i) {
  
        CHECK(i<1)<<"better have only one input blob";
    
        const Dtype* bottom_data = bottom[i]->cpu_data();
        Dtype* top_data = top[i]->mutable_cpu_data();
        Dtype* col_data = col_buffer_.mutable_cpu_data();
        Dtype* disorder_data = data_disorder_buffer_.mutable_cpu_data();
        Dtype* C_data = C_buffer_.mutable_cpu_data();
        Dtype* Gate_data = Gate_buffer_.mutable_cpu_data();
        Dtype* H_data = H_buffer_.mutable_cpu_data();
        Dtype* fc_1_data = FC_1_buffer_.mutable_cpu_data();
        const Dtype* weight = this->blobs_[0]->cpu_data();
        
        CHECK(bottom[i]->count() == data_disorder_buffer_.count());
    
        disorder_inputdata((const Dtype *)bottom_data,disorder_data,horizontal_,reverse_);
        // now disorder_data  == x, next we should do lstm from left to right, col by col
    
    
        M_ = 4 * channels_;
        N_ = num_ * col_length_;
        K_ = channels_;
        
        const int weight_count = channels_ * channels_;
        const int gate_count = M_ * N_;
        const Dtype* weight_x = weight;
        const Dtype* weight_h = weight + 4 * weight_count;
    
        for(int col=0; col < T_; col++)
        {//finish left to right lstm in this loop
            Dtype* gate_data_t = Gate_data + col*gate_count;
            Dtype* i_t = gate_data_t;
            Dtype* f_t = gate_data_t + col_count_;
            Dtype* o_t = gate_data_t + 2 * col_count_;
            Dtype* g_t = gate_data_t + 3 * col_count_;
        
            //gate(t)=W(x)*X(t)'
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
                    (Dtype)1., weight_x,disorder_data+ col* col_count_ ,
                    (Dtype)0., gate_data_t );
            
            //gate(t)+=W(h)*H(t-1)' if t >0
            if(col > 0)
            {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
                        (Dtype)1., weight_h ,H_data+ (col-1)* col_count_ ,
                        (Dtype)1., gate_data_t );
            }
            
            //gate(t)+=b
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
                    (Dtype)1.,this->blobs_[1]->cpu_data(), bias_multiplier_.cpu_data(), 
                    (Dtype)1., gate_data_t );
            
            //active gate(t)
            //sigmoid_Forward_cpu(3*col_count, i_t);
            //tanh_Forward_cpu(col_count,g_t);
            caffe_cpu_sigmoid_forward(3*col_count_,i_t,i_t);
            caffe_cpu_tanh_forward(col_count_,g_t,g_t);
    
            //C(t)=i_t .* g_t
            caffe_mul<Dtype>(col_count_, i_t,g_t,C_data + col * col_count_);
            
            //C(t) += f_t .* C(t-1)  if col >0
            if(col >0)
            {
                caffe_mul<Dtype>(col_count_, f_t,C_data + (col-1) * col_count_,fc_1_data);
                caffe_add<Dtype>(col_count_, C_data + col * col_count_, fc_1_data, C_data + col * col_count_);
            }
            
            // temp save tanh(C(t)) in fc_1_data
            //tanh_Forward_cpu(col_count,C_data + col * col_count, fc_1_data);
            caffe_cpu_tanh_forward(col_count_,C_data + col * col_count_, fc_1_data);
            //calucate fc_1_data = O(t).*tanh(C(t))
            caffe_mul<Dtype>(col_count_, o_t,fc_1_data,fc_1_data);
            
            //transpose to H(t)
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, K_,
                    (Dtype)1.,fc_1_data, identical_multiplier_.cpu_data(), 
                    (Dtype)0., H_data + col * col_count_ );
            
        }
    
        //then recover order to top data
        CHECK(top[i]->count() == H_buffer_.count());
        reorder_outputdata((const Dtype *)H_data,top_data,horizontal_,reverse_);
    
        //over!    
    }
}

template <typename Dtype>
void SpatialLstmLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
        const Dtype* weight = NULL;
        Dtype* weight_diff = NULL;

        //clear w diff and b diff
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
 
        M_ = 4 * channels_;
        N_ = num_ * col_length_;
        K_ = channels_;
        
        const int weight_count = channels_ * channels_;
        const int gate_count = M_ * N_;
        const Dtype* weight_x = weight;
        const Dtype* weight_h = weight + 4 * weight_count;
        Dtype* weight_x_diff = weight_diff;
        Dtype* weight_h_diff = weight_diff + 4 * weight_count;
  
        for (int i = 0; i < top.size(); ++i) {
        
            const Dtype* top_diff = top[i]->cpu_diff();
            Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
            Dtype* col_diff = col_buffer_.mutable_cpu_diff();
            Dtype* disorder_diff = data_disorder_buffer_.mutable_cpu_diff();
            const Dtype* disorder_data = data_disorder_buffer_.cpu_data();
            Dtype* H_diff = H_buffer_.mutable_cpu_diff();
            Dtype* C_diff = C_buffer_.mutable_cpu_diff();
            Dtype* Gate_diff = Gate_buffer_.mutable_cpu_diff();
            const Dtype* C_data = C_buffer_.cpu_data();
            const Dtype* Gate_data = Gate_buffer_.cpu_data();
            const Dtype* H_data = H_buffer_.cpu_data();
            Dtype* tanh_ct_data = FC_1_buffer_.mutable_cpu_data();
            Dtype* ht_trans_diff = FC_1_buffer_.mutable_cpu_diff();
            
            //clear diff
            caffe_set(C_buffer_.count(), Dtype(0), C_diff);
            caffe_set(Gate_buffer_.count(), Dtype(0), Gate_diff);
        
            //H(i)_diff = top_diff
            CHECK(top[i]->count() == data_disorder_buffer_.count());
            disorder_inputdata((const Dtype *)top_diff,H_diff,horizontal_,reverse_);
        
            for(int col= T_ - 1; col >= 0; col--)
            {
                Dtype* gate_diff_t = Gate_diff + col*gate_count;
                Dtype* i_t_diff = gate_diff_t;
                Dtype* f_t_diff = gate_diff_t + col_count_;
                Dtype* o_t_diff = gate_diff_t + 2 * col_count_;
                Dtype* g_t_diff = gate_diff_t + 3 * col_count_;
                Dtype* c_t_diff = C_diff + col * col_count_;
                Dtype* x_t_diff = disorder_diff + col * col_count_;
                Dtype* h_t_diff = H_diff + col * col_count_;
                
                const Dtype* gate_data_t = Gate_data + col*gate_count;
                const Dtype* i_t = gate_data_t;
                const Dtype* f_t = gate_data_t + col_count_;
                const Dtype* o_t = gate_data_t + 2 * col_count_;
                const Dtype* g_t = gate_data_t + 3 * col_count_;
                const Dtype* c_t = C_data + col * col_count_;
                const Dtype* x_t = disorder_data + col * col_count_;
                const Dtype* h_t = H_data + col * col_count_;
            
                //H(t)_diff += gate_diff(t+1)' * W_H if t < T -1
                if(col < T_ - 1)
                {
                    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_,
                            (Dtype)1.,gate_diff_t + gate_count, weight_h, 
                            (Dtype)1., h_t_diff);
                }
            
                //transpose H(t)_diff
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_, N_, K_,
                    (Dtype)1.,identical_multiplier_.cpu_data(), h_t_diff, 
                    (Dtype)0., ht_trans_diff);
                
                //O(t)_diff = H(t)_diff'.*tanh(C(t))
                //tanh_Forward_cpu(col_count,c_t, tanh_ct_data);
                caffe_cpu_tanh_forward(col_count_,c_t, tanh_ct_data);
                caffe_mul<Dtype>(col_count_, ht_trans_diff, tanh_ct_data, o_t_diff);
                
                //C(t)_diff = O(t).*H(t)_diff'.*(1-tanh(C(t))^2)
                caffe_mul<Dtype>(col_count_, o_t, ht_trans_diff, c_t_diff);
                //tanh_Backward_cpu(col_count,tanh_ct_data,c_t_diff);
                caffe_cpu_tanh_backward(col_count_,tanh_ct_data,c_t_diff,c_t_diff);
                
                //C(t)_diff  += C(t+1)_diff.*f(t+1)  if col < T-1
                if(col < T_ - 1)
                {
                    //save C(t+1)_diff.*f(t+1) in tanh_ct_data
                    caffe_mul<Dtype>(col_count_, c_t_diff + col_count_, f_t + gate_count, tanh_ct_data);
                    //then add to C(t)_diff
                    caffe_add<Dtype>(col_count_, c_t_diff, tanh_ct_data, c_t_diff);
                }
                
                //f(t)_diff = C(t)_diff.*C(t-1) if col >0
                if(col>0)
                {
                    caffe_mul<Dtype>(col_count_, c_t_diff, c_t - col_count_, f_t_diff);
                }
                
                //g(t)_diff = C(t)_diff.*i(t)
                caffe_mul<Dtype>(col_count_, c_t_diff, i_t, g_t_diff);
                
                //i(t)_diff = C(t)_diff.*g(t)
                caffe_mul<Dtype>(col_count_, c_t_diff, g_t, i_t_diff);
                
                //active backward f g i o
                //sigmoid_Backward_cpu(3*col_count, i_t, i_t_diff);
                //tanh_Backward_cpu(col_count, g_t, g_t_diff);
                caffe_cpu_sigmoid_backward(3*col_count_, i_t, i_t_diff,i_t_diff);
                caffe_cpu_tanh_backward(col_count_, g_t, g_t_diff,g_t_diff);
                
                if(this->param_propagate_down_[0])
                {
                    //W_X_diff += gate_diff(t) * x(t)
                    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_,
                        (Dtype)1.,gate_diff_t, x_t, 
                        (Dtype)1., weight_x_diff);
                    
                    //W_H_diff += gate_diff(t) * h(t-1) if t >0
                    if(col > 0)
                    {
                        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_,
                            (Dtype)1.,gate_diff_t, h_t - col_count_, 
                            (Dtype)1., weight_h_diff);
                    }
                }
                
                if(propagate_down[i])
                {
                    //X(t)_diff = gate_diff(t)' * W_X
                    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_,
                            (Dtype)1.,gate_diff_t, weight_x, 
                            (Dtype)0., x_t_diff);
                }
                if(this->param_propagate_down_[1])
                {
                    //b_diff += gate_diff(t) * 1_col
                    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, 1, N_,
                            (Dtype)1.,gate_diff_t, bias_multiplier_.cpu_data(), 
                            (Dtype)1., bias_diff);
                }
            
            }
            
            if(propagate_down[i])
            {//finish bottom diff
                // recover order to bottom diff
                CHECK(bottom[i]->count() == data_disorder_buffer_.count());
                reorder_outputdata((const Dtype *)disorder_diff,bottom_diff,horizontal_,reverse_);
            }
                
        }
}

#ifdef CPU_ONLY
STUB_GPU(SpatialLstmLayer);
#endif

INSTANTIATE_CLASS(SpatialLstmLayer);
REGISTER_LAYER_CLASS(SpatialLstm);

}  // namespace caffe
