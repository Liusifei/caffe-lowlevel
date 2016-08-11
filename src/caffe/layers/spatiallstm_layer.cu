#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SpatialLstmLayer<Dtype>::disorder_gpu_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse)
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
void SpatialLstmLayer<Dtype>::reorder_gpu_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse)
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
void SpatialLstmLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    for (int i = 0; i < bottom.size(); ++i) {
  
        CHECK(i<1)<<"better have only one input blob";
    //LOG(INFO)<<"in splstm fp ";
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* top_data = top[i]->mutable_gpu_data();
        Dtype* col_data = col_buffer_.mutable_gpu_data();
        Dtype* disorder_data = data_disorder_buffer_.mutable_gpu_data();
        Dtype* C_data = C_buffer_.mutable_gpu_data();
        Dtype* Gate_data = Gate_buffer_.mutable_gpu_data();
        Dtype* H_data = H_buffer_.mutable_gpu_data();
        Dtype* fc_1_data = FC_1_buffer_.mutable_gpu_data();
        const Dtype* weight = this->blobs_[0]->gpu_data();
        
        CHECK(bottom[i]->count() == data_disorder_buffer_.count());
    
        disorder_gpu_inputdata((const Dtype *)bottom_data,disorder_data,horizontal_,reverse_);
        // now disorder_data  == x, next we should do lstm from left to right, col by col
    
    
        M_ = 4 * channels_;
        N_ = num_ * col_length_;
        K_ = channels_;
        
        const int weight_count = channels_ * channels_;
        const int gate_count = M_ * N_;
        const Dtype* weight_x = weight;
        const Dtype* weight_h = weight + 4 * weight_count;
    //LOG(INFO)<<"in splstm fp before for loop";
        for(int col=0; col < T_; col++)
        {//finish left to right lstm in this loop
            Dtype* gate_data_t = Gate_data + col*gate_count;
            Dtype* i_t = gate_data_t;
            Dtype* f_t = gate_data_t + col_count_;
            Dtype* o_t = gate_data_t + 2 * col_count_;
            Dtype* g_t = gate_data_t + 3 * col_count_;
        
            //gate(t)=W(x)*X(t)'
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
                    (Dtype)1., weight_x,disorder_data+ col* col_count_ ,
                    (Dtype)0., gate_data_t );
            
            //gate(t)+=W(h)*H(t-1)' if t >0
            if(col > 0)
            {
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
                        (Dtype)1., weight_h ,H_data+ (col-1)* col_count_ ,
                        (Dtype)1., gate_data_t );
            }
            
            //gate(t)+=b
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
                    (Dtype)1.,this->blobs_[1]->gpu_data(), bias_multiplier_.gpu_data(), 
                    (Dtype)1., gate_data_t );
            
            //active gate(t)
            //sigmoid_Forward_gpu(3*col_count, i_t);
            //tanh_Forward_gpu(col_count,g_t);
            caffe_gpu_sigmoid_forward(3*col_count_,i_t,i_t);
            caffe_gpu_tanh_forward(col_count_,g_t,g_t);
    
            //C(t)=i_t .* g_t
            caffe_gpu_mul<Dtype>(col_count_, i_t,g_t,C_data + col * col_count_);
            
            //C(t) += f_t .* C(t-1)  if col >0
            if(col >0)
            {
                caffe_gpu_mul<Dtype>(col_count_, f_t,C_data + (col-1) * col_count_,fc_1_data);
                caffe_gpu_add<Dtype>(col_count_, C_data + col * col_count_, fc_1_data, C_data + col * col_count_);
            }
            
            // temp save tanh(C(t)) in fc_1_data
            //tanh_Forward_gpu(col_count,C_data + col * col_count, fc_1_data);
            caffe_gpu_tanh_forward(col_count_,C_data + col * col_count_, fc_1_data);
            //calucate fc_1_data = O(t).*tanh(C(t))
            caffe_gpu_mul<Dtype>(col_count_, o_t,fc_1_data,fc_1_data);
            
            //transpose to H(t)
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, K_,
                    (Dtype)1.,fc_1_data, identical_multiplier_.gpu_data(), 
                    (Dtype)0., H_data + col * col_count_ );
            
        }
    //LOG(INFO)<<"in splstm fp after for loop ";
        //then recover order to top data
        CHECK(top[i]->count() == H_buffer_.count());
        reorder_gpu_outputdata((const Dtype *)H_data,top_data,horizontal_,reverse_);
    //LOG(INFO)<<"in splstm fp over ";
        //over!    
    }
}

template <typename Dtype>
void SpatialLstmLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* weight = NULL;
        Dtype* weight_diff = NULL;

        //clear w diff and b diff
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
        
            const Dtype* top_diff = top[i]->gpu_diff();
            Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
            Dtype* col_diff = col_buffer_.mutable_gpu_diff();
            Dtype* disorder_diff = data_disorder_buffer_.mutable_gpu_diff();
            const Dtype* disorder_data = data_disorder_buffer_.gpu_data();
            Dtype* H_diff = H_buffer_.mutable_gpu_diff();
            Dtype* C_diff = C_buffer_.mutable_gpu_diff();
            Dtype* Gate_diff = Gate_buffer_.mutable_gpu_diff();
            const Dtype* C_data = C_buffer_.gpu_data();
            const Dtype* Gate_data = Gate_buffer_.gpu_data();
            const Dtype* H_data = H_buffer_.gpu_data();
            Dtype* tanh_ct_data = FC_1_buffer_.mutable_gpu_data();
            Dtype* ht_trans_diff = FC_1_buffer_.mutable_gpu_diff();
            
            //clear diff
            caffe_gpu_set(C_buffer_.count(), Dtype(0), C_diff);
            caffe_gpu_set(Gate_buffer_.count(), Dtype(0), Gate_diff);
        
            //H(i)_diff = top_diff
            CHECK(top[i]->count() == data_disorder_buffer_.count());
            disorder_gpu_inputdata((const Dtype *)top_diff,H_diff,horizontal_,reverse_);
        
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
                    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_,
                            (Dtype)1.,gate_diff_t + gate_count, weight_h, 
                            (Dtype)1., h_t_diff);
                }
            
                //transpose H(t)_diff
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_, N_, K_,
                    (Dtype)1.,identical_multiplier_.gpu_data(), h_t_diff, 
                    (Dtype)0., ht_trans_diff);
                
                //O(t)_diff = H(t)_diff'.*tanh(C(t))
                //tanh_Forward_gpu(col_count,c_t, tanh_ct_data);
                caffe_gpu_tanh_forward(col_count_,c_t, tanh_ct_data);
                caffe_gpu_mul<Dtype>(col_count_, ht_trans_diff, tanh_ct_data, o_t_diff);
                
                //C(t)_diff = O(t).*H(t)_diff'.*(1-tanh(C(t))^2)
                caffe_gpu_mul<Dtype>(col_count_, o_t, ht_trans_diff, c_t_diff);
                //tanh_Backward_gpu(col_count,tanh_ct_data,c_t_diff);
                caffe_gpu_tanh_backward(col_count_,tanh_ct_data,c_t_diff,c_t_diff);
                
                //C(t)_diff  += C(t+1)_diff.*f(t+1)  if col < T-1
                if(col < T_ - 1)
                {
                    //save C(t+1)_diff.*f(t+1) in tanh_ct_data
                    caffe_gpu_mul<Dtype>(col_count_, c_t_diff + col_count_, f_t + gate_count, tanh_ct_data);
                    //then add to C(t)_diff
                    caffe_gpu_add<Dtype>(col_count_, c_t_diff, tanh_ct_data, c_t_diff);
                }
                
                //f(t)_diff = C(t)_diff.*C(t-1) if col >0
                if(col>0)
                {
                    caffe_gpu_mul<Dtype>(col_count_, c_t_diff, c_t - col_count_, f_t_diff);
                }
                
                //g(t)_diff = C(t)_diff.*i(t)
                caffe_gpu_mul<Dtype>(col_count_, c_t_diff, i_t, g_t_diff);
                
                //i(t)_diff = C(t)_diff.*g(t)
                caffe_gpu_mul<Dtype>(col_count_, c_t_diff, g_t, i_t_diff);
                
                //active backward f g i o
                //sigmoid_Backward_gpu(3*col_count, i_t, i_t_diff);
                //tanh_Backward_gpu(col_count, g_t, g_t_diff);
                caffe_gpu_sigmoid_backward(3*col_count_, i_t, i_t_diff,i_t_diff);
                caffe_gpu_tanh_backward(col_count_, g_t, g_t_diff,g_t_diff);
                
                if(this->param_propagate_down_[0])
                {
                    //W_X_diff += gate_diff(t) * x(t)
                    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_,
                        (Dtype)1.,gate_diff_t, x_t, 
                        (Dtype)1., weight_x_diff);
                    
                    //W_H_diff += gate_diff(t) * h(t-1) if t >0
                    if(col > 0)
                    {
                        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_,
                            (Dtype)1.,gate_diff_t, h_t - col_count_, 
                            (Dtype)1., weight_h_diff);
                    }
                }
                
                if(propagate_down[i])
                {
                    //X(t)_diff = gate_diff(t)' * W_X
                    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_,
                            (Dtype)1.,gate_diff_t, weight_x, 
                            (Dtype)0., x_t_diff);
                }
                if(this->param_propagate_down_[1])
                {
                    //b_diff += gate_diff(t) * 1_col
                    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, 1, N_,
                            (Dtype)1.,gate_diff_t, bias_multiplier_.gpu_data(), 
                            (Dtype)1., bias_diff);
                }
            
            }
            
            if(propagate_down[i])
            {//finish bottom diff
                // recover order to bottom diff
                CHECK(bottom[i]->count() == data_disorder_buffer_.count());
                reorder_gpu_outputdata((const Dtype *)disorder_diff,bottom_diff,horizontal_,reverse_);
            }
                
        }
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialLstmLayer);

}  // namespace caffe
