#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ExpandlabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    maxlabel = this->layer_param_.expandlabel_param().maxlabel();
    this->duplicate_channels_ = this->layer_param_.expandlabel_param().duplicate_channels();
    int channels = bottom[0]->channels();
    CHECK(channels == 1) << "input of ExpandlabelLayer must have only 1 channels, but here get "<<channels;
     
    int num = bottom[0]->num();
    int height = bottom[0]->height();
    int width = bottom[0]->width();
    
    
    
}

template <typename Dtype>
void ExpandlabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      
            int num = bottom[0]->num();
            int height = bottom[0]->height();
            int width = bottom[0]->width();
            
            switch (this->layer_param_.expandlabel_param().type()) 
            {
                case ExpandlabelParameter_Type_EXPAND:
                    top[0]->Reshape(num, maxlabel+1,height, width);
                    break;
                case ExpandlabelParameter_Type_DUPLICATE:
                    CHECK(this->duplicate_channels_ > 1);
                    top[0]->Reshape(num, this->duplicate_channels_,height, width);
                    break;
                default:
                    LOG(FATAL) << "Unknown type method.";
            }
      
 ;
}

template <typename Dtype>
void ExpandlabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      
        const int count = top[0]->count();
        Dtype* top_data = top[0]->mutable_cpu_data();
        

        int num = bottom[0]->num();
        int height = bottom[0]->height();
        int width = bottom[0]->width();
        int spatial_dim = height * width;
        
        int dim = top[0]->count() / num;
        
        CHECK(dim == spatial_dim * (maxlabel +1))<<"ExpandlabelLayer dim == spatial_dim * (maxlabel +1)";
        
        const Dtype* label = bottom[0]->cpu_data();
        
        
        switch (this->layer_param_.expandlabel_param().type()) 
        {
            case ExpandlabelParameter_Type_EXPAND:
                caffe_set(count, Dtype(0), top_data);
                for (int i = 0; i < num; ++i) {
                    for (int j = 0; j < spatial_dim; j++) {
                     
                        int lb = static_cast<int>(label[i * spatial_dim + j]);
                        if(lb > maxlabel)
                        {
                            CHECK(false) << "max label was set to be "<<maxlabel<<", but here get "<<lb<<"in ExpandlabelLayer";
                        }
                        
                        top_data[i * dim + lb * spatial_dim + j] = Dtype(1);
                    
                    }
                }
                break;
            case ExpandlabelParameter_Type_DUPLICATE:
                for (int i = 0; i < this->duplicate_channels_; ++i) 
                {
                    caffe_copy(bottom[0]->count(),bottom[0]->cpu_data(),top_data + i*bottom[0]->count());
                }
                break;
            default:
                LOG(FATAL) << "Unknown type method.";
        }
            
            
        
        
}

template <typename Dtype>
void ExpandlabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      
        Dtype* top_diff = top[0]->mutable_cpu_diff();
        
        switch (this->layer_param_.expandlabel_param().type()) 
        {
            case ExpandlabelParameter_Type_EXPAND:
                caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
                break;
            case ExpandlabelParameter_Type_DUPLICATE:
                caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
                
                for (int i = 0; i < this->duplicate_channels_; ++i) 
                {
                    caffe_add(bottom[0]->count(),top_diff + i * bottom[0]->count(),bottom[0]->cpu_diff(),bottom[0]->mutable_cpu_diff());
                }
                break;
            default:
                LOG(FATAL) << "Unknown type method.";
        }
}

#ifdef CPU_ONLY
STUB_GPU(ExpandlabelLayer);
#endif

INSTANTIATE_CLASS(ExpandlabelLayer);
REGISTER_LAYER_CLASS(Expandlabel);

}  // namespace caffe
