#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void GateLstmLayer<Dtype>::disorder_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse)
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
void GateLstmLayer<Dtype>::reorder_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse)
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
void GateLstmLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
        
        channels_ = bottom[0]->channels();
        height_out_ = bottom[0]->height();
        width_out_ = bottom[0]->width();

        num_output_ = this->layer_param_.gatelstm_param().num_output();
        horizontal_ = this->layer_param_.gatelstm_param().horizontal();
        reverse_ = this->layer_param_.gatelstm_param().reverse();
        bias_term_ = this->layer_param_.gatelstm_param().bias_term();
        restrict_w_ = this->layer_param_.gatelstm_param().restrict_w();
        
        CHECK(bottom.size() ==2 )<<"bottom size can only be 2";   

        int gate_channels = bottom[1]->channels();
        //CHECK(gate_channels_ == 1 || gate_channels_ == num_output_)<<"gate channels can only be 1 or equal to top channels";
        CHECK(gate_channels == num_output_)<<"gate channels can only equal to top channels";
        
        if (this->blobs_.size() > 0) 
        {
            LOG(INFO) << "Skipping parameter initialization";
        } 
        else {
            if (bias_term_)
            {
                this->blobs_.resize(3);
            } 
            else
            {
                this->blobs_.resize(2);
            }

            //Wx 
            this->blobs_[0].reset(new Blob<Dtype>(1, 1, 3 * num_output_, channels_));
            shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
                this->layer_param_.gatelstm_param().weight_filler()));
            weight_filler->Fill(this->blobs_[0].get());
            
            //Wh
            this->blobs_[1].reset(new Blob<Dtype>(1, 1, 3 * num_output_, num_output_));
            weight_filler->Fill(this->blobs_[1].get());

            if (bias_term_)
            {
                //b  B
                this->blobs_[2].reset(new Blob<Dtype>(1, 2, 3 * num_output_,1));
                shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                    this->layer_param_.gatelstm_param().bias_filler()));
                bias_filler->Fill(this->blobs_[2].get());
            }
        }

        this->param_propagate_down_.resize(this->blobs_.size(), true);
        
        identical_multiplier_.Reshape(1, 1, num_output_,num_output_);
        Dtype* id_data = identical_multiplier_.mutable_cpu_data();
        for (int h=0;h<num_output_;h++)
        {
            for(int w=0;w<num_output_;w++)
            {
                if(w == h)
                {
                    id_data[h*num_output_ + w] = Dtype(1);
                }
                else
                {
                    id_data[h*num_output_ + w] = Dtype(0);
                }
            }
        }
}

template <typename Dtype>
void GateLstmLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      
        num_ = bottom[0]->num();
        height_ = bottom[0]->height();
        width_ = bottom[0]->width();
        CHECK_EQ(bottom[0]->channels(), channels_) << "Input channels incompatible ";
        CHECK_EQ(height_, height_out_) << "Input height incompatible ";
        CHECK_EQ(width_, width_out_) << "Input width incompatible ";

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
        x_col_count_ = col_length_ * num_ * channels_;
        h_col_count_ = col_length_ * num_ * num_output_;
    
        // TODO: generalize to handle inputs of different shapes.
        for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) 
        {
            CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
            
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

        
        L_buffer_.Reshape(3 * num_ , num_output_ , height_ , width_);
        P_buffer_.Reshape(3 * num_ , num_output_ , height_ , width_);
        C_buffer_.Reshape(num_ , num_output_ , height_ , width_);
        G_buffer_.Reshape(num_ , num_output_ , height_ , width_);
        
        H_buffer_.Reshape(num_ , num_output_ , height_, width_);
        X_buffer_.Reshape(num_ , channels_ , height_, width_);
        
        GL_buffer_.Reshape(1 , 1 , 3 * num_output_ , num_ * col_length_);
        Trans_buffer_.Reshape(1 , 1 , num_output_ , num_ * col_length_);
        Ct_active_buffer_.Reshape(1 , 1 , num_output_ , num_ * col_length_);
        
        if (bias_term_) 
        {
            bias_multiplier_.Reshape(1, 1, 1, num_ * col_length_);
            caffe_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_cpu_data());
        }
}

template <typename Dtype>
void GateLstmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
    if(restrict_w_)
    {
        caffe_bound(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(), Dtype(-1.0), Dtype(1.0), this->blobs_[0]->mutable_cpu_data());
        caffe_bound(this->blobs_[1]->count(), this->blobs_[1]->cpu_data(), Dtype(-1.0), Dtype(1.0), this->blobs_[1]->mutable_cpu_data());
    }
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_gate = bottom[1]->cpu_data();
    const Dtype* W_x = this->blobs_[0]->cpu_data();
    const Dtype* W_h = this->blobs_[1]->cpu_data();
    const Dtype* bias_data = this->blobs_[2]->cpu_data();
       
    Dtype* X_data = X_buffer_.mutable_cpu_data();
    Dtype* H_data = H_buffer_.mutable_cpu_data();
    Dtype* G_data = G_buffer_.mutable_cpu_data();
    Dtype* L_data = L_buffer_.mutable_cpu_data();
    Dtype* P_data = P_buffer_.mutable_cpu_data();
    Dtype* C_data = C_buffer_.mutable_cpu_data();
    Dtype* GL_mid_data = GL_buffer_.mutable_cpu_data();
    Dtype* Trans_data = Trans_buffer_.mutable_cpu_data();
    Dtype* ct_active = Ct_active_buffer_.mutable_cpu_data();
       
    //get X_data
    disorder_inputdata((const Dtype *)bottom_data,X_data,horizontal_,reverse_);

    //get gate data
    disorder_inputdata((const Dtype *)bottom_gate,G_data,horizontal_,reverse_);
    
    M_ = 3 * num_output_;
    N_ = num_ * col_length_;
    K_x_ = channels_;
    K_h_ = num_output_;
    
    int L_count = M_ * N_;
    int P_count = M_ * N_;
    int G_count = h_col_count_;
    int C_count = h_col_count_;
    int bias_count = M_;
    
    const Dtype* b1_data =  bias_data;
    const Dtype* B2_data =  bias_data + bias_count;
    
    for(int t=0; t < T_; t++)
    {//finish left to right gate lstm in this loop
    
        Dtype* L_t = L_data + t * L_count;
        Dtype* P_t = P_data + t * P_count;
        Dtype* G_t = G_data + t * G_count;
        Dtype* C_t = C_data + t * C_count;
        Dtype* i_t = P_t;
        Dtype* o_t = P_t + h_col_count_;
        Dtype* u_t = P_t + 2 * h_col_count_;  

        //L(t) = b
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
                (Dtype)1.,b1_data, bias_multiplier_.cpu_data(), 
                (Dtype)0., L_t );
        
        //L(t) += W_h * H(t-1)' if t > 0
        if(t > 0)
        {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_h_,
                (Dtype)1.,W_h, H_data+ (t-1)* h_col_count_, 
                (Dtype)1., L_t );
        }
        
        //P(t) = B
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
                (Dtype)1.,B2_data, bias_multiplier_.cpu_data(), 
                (Dtype)0., P_t );
        
        //P(t) += W_x * X(t)'
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_x_,
                (Dtype)1.,W_x, X_data + t * x_col_count_, 
                (Dtype)1., P_t );
        
        //transpose G(t) to  trans_data , now trans_data is G(t)'
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_h_, N_, K_h_,
                    (Dtype)1.,identical_multiplier_.cpu_data(), G_t, 
                    (Dtype)0., Trans_data);
                    
        //P(t) += G(t)' .* L(t)
        caffe_mul<Dtype>(h_col_count_, Trans_data, L_t, GL_mid_data);
        caffe_mul<Dtype>(h_col_count_, Trans_data, L_t + h_col_count_, GL_mid_data + h_col_count_);
        caffe_mul<Dtype>(h_col_count_, Trans_data, L_t + 2 * h_col_count_, GL_mid_data + 2 * h_col_count_);
        caffe_add<Dtype>(P_count, P_t, GL_mid_data , P_t );
        
        //active P(t) --> i o u
        caffe_cpu_sigmoid_forward(2*h_col_count_,i_t,i_t);
        caffe_cpu_tanh_forward(h_col_count_,u_t,u_t);
        
        //C(t) = i(t) .* u(t)
        caffe_mul<Dtype>(h_col_count_, i_t, u_t, C_t);
        
        //C(t) += G(t)' .* C(t-1)  if t >0 
        if(t>0)
        {
            //temp save G(t)' .* C(t-1)  in GL_mid_data
            caffe_mul<Dtype>(C_count, Trans_data, C_data + (t-1)*C_count, GL_mid_data);
            //then add to C(t)
            caffe_add<Dtype>(C_count, GL_mid_data, C_t , C_t );
        }
        
        //active C(t)
        caffe_cpu_tanh_forward(C_count, C_t, ct_active);
        
        //save  o(t).* active[C(t)]  into trans_data
        caffe_mul<Dtype>(C_count, o_t, ct_active, Trans_data);

        //transpose trans_data to H(t)
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_h_, K_h_,
                    (Dtype)1.,Trans_data, identical_multiplier_.cpu_data(),
                    (Dtype)0., H_data + t * h_col_count_);
    }
    //then recover order to top data
    reorder_outputdata((const Dtype *)H_data,top[0]->mutable_cpu_data(),horizontal_,reverse_);

    //over!    
    
}

template <typename Dtype>
void GateLstmLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    
    M_ = 3 * num_output_;
    N_ = num_ * col_length_;
    K_x_ = channels_;
    K_h_ = num_output_;
    
    int L_count = M_ * N_;
    int P_count = M_ * N_;
    int G_count = h_col_count_;
    int C_count = h_col_count_;
    int bias_count = M_;
     
    const Dtype* W_x = NULL;
    Dtype* W_x_diff = NULL;
    const Dtype* W_h = NULL;
    Dtype* W_h_diff = NULL;

    //clear w diff and b diff
    if (this->param_propagate_down_[0]) {
        W_x = this->blobs_[0]->cpu_data();
        W_x_diff = this->blobs_[0]->mutable_cpu_diff();
        caffe_set(this->blobs_[0]->count(), Dtype(0), W_x_diff);
    }
    if (this->param_propagate_down_[1]) {
        W_h = this->blobs_[1]->cpu_data();
        W_h_diff = this->blobs_[1]->mutable_cpu_diff();
        caffe_set(this->blobs_[1]->count(), Dtype(0), W_h_diff);
    }
    Dtype* bias_diff = NULL;
    Dtype* b1_diff = NULL;
    Dtype* B2_diff = NULL;
    if (bias_term_ && this->param_propagate_down_[2]) {
        bias_diff = this->blobs_[2]->mutable_cpu_diff();
        caffe_set(this->blobs_[2]->count(), Dtype(0), bias_diff);
        b1_diff = bias_diff;
        B2_diff = bias_diff + bias_count;
        
    } 
    
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* C_data = C_buffer_.cpu_data();
    const Dtype* P_data = P_buffer_.cpu_data();
    const Dtype* L_data = L_buffer_.cpu_data();
    const Dtype* G_data = G_buffer_.cpu_data();
    const Dtype* X_data = X_buffer_.cpu_data();
    const Dtype* H_data = H_buffer_.cpu_data();
    
    //Dtype* bottom_diff = bottom[0]->mutable_cpu_diff(); 
    Dtype* H_diff = H_buffer_.mutable_cpu_diff();
    Dtype* X_diff = X_buffer_.mutable_cpu_diff();
    Dtype* C_diff = C_buffer_.mutable_cpu_diff();
    Dtype* P_diff = P_buffer_.mutable_cpu_diff();
    Dtype* L_diff = L_buffer_.mutable_cpu_diff();
    Dtype* G_diff = G_buffer_.mutable_cpu_diff();
    
    Dtype* Trans_diff = Trans_buffer_.mutable_cpu_diff();
    Dtype* Trans_G = Trans_buffer_.mutable_cpu_data();
    Dtype* ct_active = Ct_active_buffer_.mutable_cpu_data();
    Dtype* temp_PL_diff = GL_buffer_.mutable_cpu_diff();
  
    //clear diff
    //??

    //H(i)_diff = top_diff
    disorder_inputdata((const Dtype *)top_diff,H_diff,horizontal_,reverse_);

    for(int t= T_ - 1; t >= 0; t--)
    {//finish right to left gate lstm BP in this loop
        const Dtype* G_t = G_data + t*G_count;
        const Dtype* C_t = C_data + t*C_count;
        const Dtype* P_t = P_data + t*P_count;
        const Dtype* L_t = L_data + t*L_count;
        const Dtype* i_t = P_t ;
        const Dtype* o_t = P_t + h_col_count_;
        const Dtype* u_t = P_t + 2*h_col_count_;
        
        Dtype* L_t_diff = L_diff + t*L_count;
        Dtype* G_t_diff = G_diff + t*G_count;
        Dtype* C_t_diff = C_diff + t*C_count;
        Dtype* P_t_diff = P_diff + t*P_count;
        Dtype* i_t_diff = P_t_diff;
        Dtype* o_t_diff = P_t_diff + h_col_count_;
        Dtype* u_t_diff = P_t_diff + 2*h_col_count_;

        //H(t)_diff += L(t+1)_diff' * W_h  if t < T-1
        if(t < T_-1)
        {
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_h_, M_,
                    (Dtype)1., L_diff + (t+1)*L_count,  W_h,
                    (Dtype)1., H_diff + t * h_col_count_);
        }
        
        //transpose H(t)_diff, now Trans_diff is H(t)_diff'
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_h_, N_, K_h_,
                    (Dtype)1., identical_multiplier_.cpu_data(), H_diff + t * h_col_count_ ,
                    (Dtype)0., Trans_diff);
        
        
        
        //O(t)_diff = H(t)_diff' .*  active[C(t)]
        caffe_cpu_tanh_forward(C_count, C_t, ct_active);
        caffe_mul<Dtype>(h_col_count_, Trans_diff, ct_active, o_t_diff);
        
        //C(t)_diff = H(t)_diff' .* O(t) ==>> activeback[C(t)]
        caffe_mul<Dtype>(h_col_count_, Trans_diff, o_t, C_t_diff);
        caffe_cpu_tanh_backward(h_col_count_,ct_active,C_t_diff,C_t_diff);
        
        //transpose G(t+1), now Trans_G is G(t+1)'
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_h_, N_, K_h_,
                    (Dtype)1., identical_multiplier_.cpu_data(), G_data + (t+1) * G_count ,
                    (Dtype)0., Trans_G);
                    
        //C(t)_diff  += C(t+1)_diff .* G(t+1)'  if t < T-1
        if(t < T_-1)
        {
            //save C(t+1)_diff .* G(t+1)' in ct_active
            caffe_mul<Dtype>(C_count, C_diff + (t+1)*C_count, Trans_G, ct_active);
            // then add to C(t)_diff
            caffe_add<Dtype>(C_count, ct_active, C_t_diff, C_t_diff);
        }
        
        //i(t)_diff = C(t)_diff .* u(t)
        caffe_mul<Dtype>(h_col_count_, C_t_diff, u_t, i_t_diff);
        
        //u(t)_diff = C(t)_diff .* i(t)
        caffe_mul<Dtype>(h_col_count_, C_t_diff, i_t, u_t_diff);
        
        //active back P(t)_diff  --> i, o, u
        caffe_cpu_sigmoid_backward(2*h_col_count_, i_t, i_t_diff,i_t_diff);
        caffe_cpu_tanh_backward(h_col_count_, u_t, u_t_diff,u_t_diff);
        
        //transpose G(t), now Trans_G is G(t)'
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_h_, N_, K_h_,
                    (Dtype)1., identical_multiplier_.cpu_data(), G_t ,
                    (Dtype)0., Trans_G);
                    
        //L(t)_diff = P(t)_diff .* G(t)'   , need expand 3 
        caffe_mul<Dtype>(h_col_count_, P_t_diff, Trans_G, L_t_diff);
        caffe_mul<Dtype>(h_col_count_, P_t_diff + h_col_count_, Trans_G, L_t_diff + h_col_count_);
        caffe_mul<Dtype>(h_col_count_, P_t_diff + 2*h_col_count_, Trans_G, L_t_diff + 2*h_col_count_);
        
        if(propagate_down[1])
        {
            //G(t)_diff' = P(t)_diff .* L(t),   need compress, Trans_G G(t)_diff' now
            caffe_mul<Dtype>(P_count, P_t_diff, L_t, temp_PL_diff);
            caffe_add<Dtype>(h_col_count_, temp_PL_diff, temp_PL_diff + h_col_count_, Trans_G);
            caffe_add<Dtype>(h_col_count_, temp_PL_diff + 2*h_col_count_, Trans_G, Trans_G);
            
            //G(t)_diff' += C(t)_diff .*C(t-1) if t > 0
            if(t>0)
            {
                //save C(t)_diff .*C(t-1) in temp_PL_diff 
                caffe_mul<Dtype>(h_col_count_, C_t_diff, C_data + (t-1)*C_count, temp_PL_diff);
                //then add to Trans_G
                caffe_add<Dtype>(h_col_count_, temp_PL_diff, Trans_G, Trans_G);
            }
            
            // transpose temp_G(t)_diff' to G(t)_diff
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_h_, K_h_,
                        (Dtype)1.,Trans_G, identical_multiplier_.cpu_data(),
                        (Dtype)0., G_t_diff);
        }          
        //W_x_diff += P(t)_diff * X(t)
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_x_, N_,
                    (Dtype)1., P_t_diff, X_data + t * x_col_count_,
                    (Dtype)1., W_x_diff);
        
        //W_h_diff += L(t)_diff * H(t-1) if t >0
        if(t>0)
        {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_h_, N_,
                        (Dtype)1., L_t_diff, H_data + (t-1) * h_col_count_,
                        (Dtype)1., W_h_diff);
        }
        
        if(propagate_down[0])
        {
            //X(t)_diff = P(t)_diff' * W_x
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_x_, M_,
                        (Dtype)1., P_t_diff, W_x,
                        (Dtype)0., X_diff + t * x_col_count_);     
        }
        //b(t)_diff += L(t)_diff
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, 1, N_,
                           (Dtype)1.,L_t_diff, bias_multiplier_.cpu_data(), 
                           (Dtype)1., b1_diff);

        //B(t)_diff += P(t)_diff
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, 1, N_,
                           (Dtype)1.,P_t_diff, bias_multiplier_.cpu_data(), 
                           (Dtype)1., B2_diff);
    }
    if(propagate_down[0])
    {
        reorder_outputdata((const Dtype *)X_diff,bottom[0]->mutable_cpu_diff(),horizontal_,reverse_);
    }
    if(propagate_down[1])
    {
        reorder_outputdata((const Dtype *)G_diff,bottom[1]->mutable_cpu_diff(),horizontal_,reverse_);
    }
}

#ifdef CPU_ONLY
STUB_GPU(GateLstmLayer);
#endif

INSTANTIATE_CLASS(GateLstmLayer);
REGISTER_LAYER_CLASS(GateLstm);

}  // namespace caffe
