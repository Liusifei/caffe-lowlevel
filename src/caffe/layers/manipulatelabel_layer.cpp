#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ManipulatelabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
    horizontal_ = this->layer_param_.manipulatelabel_param().horizontal();
    edgerange_ = this->layer_param_.manipulatelabel_param().edgerange();
    maxlabel_ = this->layer_param_.manipulatelabel_param().maxlabel();

}

template <typename Dtype>
void ManipulatelabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      
        int num = bottom[0]->num();
        int channels = bottom[0]->channels();
        int height = bottom[0]->height();
        int width = bottom[0]->width();
        
        switch (this->layer_param_.manipulatelabel_param().type()) 
        {
            case ManipulatelabelParameter_Type_EDGE:
                top[0]->Reshape(num, channels,height, width);
                break;
            case ManipulatelabelParameter_Type_EXPAND:
                
                CHECK(channels == 1) << "input of ManipulatelabelLayer (type:expand) must have only 1 channels, but here get "<<channels;
                top[0]->Reshape(num, maxlabel_+1,height, width);
                break;
            default:
                LOG(FATAL) << "Unknown type method.";
        }
}

template <typename Dtype>
void ManipulatelabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      
        int num = bottom[0]->num();
        int channels = bottom[0]->channels();
        int height = bottom[0]->height();
        int width = bottom[0]->width();
        const int count = top[0]->count();
        Dtype* top_data = top[0]->mutable_cpu_data();
        caffe_set(count, Dtype(0), top_data);
        const Dtype* label = bottom[0]->cpu_data();
        int spatial_dim = height * width;        
        int dim = top[0]->count() / num; 
        
     
        switch (this->layer_param_.manipulatelabel_param().type()) 
        {
            case ManipulatelabelParameter_Type_EDGE:
                CHECK(edgerange_>0)<<"edge range must bigger than 0.";
                int new_h,new_w;
                for(int n=0;n<num;n++)
                    for(int c=0;c<channels;c++)
                        for(int h=0;h<height;h++)
                            for(int w=0;w<width;w++)
                            {
                                Dtype v = label[n*dim + c*spatial_dim + h*width + w];
                                bool isedge=false;
                                for(int r = -1*edgerange_ ; r <= edgerange_;r++)
                                {
                                    if(horizontal_)
                                    {
                                        new_h = h;
                                        new_w = w + r;
                                    }
                                    else
                                    {
                                        new_h = h + r;
                                        new_w = w;
                                    }
                                    if(new_h>=0 && new_h <height && new_w>=0 && new_w < width)
                                    {
                                        if(label[n*dim + c*spatial_dim + new_h*width + new_w] != v)
                                        {
                                            isedge = true;
                                            break;
                                        }
                                    }
                                }
                                if(isedge)
                                    top_data[n*dim + c*spatial_dim + h*width + w] = 1;
                            }

                break;
            case ManipulatelabelParameter_Type_EXPAND:
                CHECK(dim == spatial_dim * (maxlabel_ +1))<<"ManipulatelabelLayer dim == spatial_dim * (maxlabel_ +1)";
                for (int i = 0; i < num; ++i) {
                    for (int j = 0; j < spatial_dim; j++) {            
                        int lb = static_cast<int>(label[i * spatial_dim + j]);
                        if(lb > maxlabel_)
                        {
                            CHECK(false) << "max label was set to be "<<maxlabel_<<", but here get "<<lb<<"in ExpandlabelLayer";
                        }                      
                        top_data[i * dim + lb * spatial_dim + j] = Dtype(1);
                    }
                }
                break;
            default:
                LOG(FATAL) << "Unknown type method.";
        }
/*
        FILE * fp =NULL;
        fp=fopen("bottom.txt","r");
          if(fp == NULL)
          {
              fp=fopen("bottom.txt","w");
              const Dtype* inputdata = bottom[0]->cpu_data();
              int count = bottom[0]->count();
              for(int tempi=0;tempi<count;tempi++)
              {
                  fprintf(fp,"%f\n",(float)inputdata[tempi]);
              }
              fclose(fp); 
          }
          else
          {
           fclose(fp);
          }        
         
        fp=fopen("top.txt","r");
          if(fp == NULL)
          {
              fp=fopen("top.txt","w");
              const Dtype* inputdata = top[0]->cpu_data();
              int count = top[0]->count();
              for(int tempi=0;tempi<count;tempi++)
              {
                  fprintf(fp,"%f\n",(float)inputdata[tempi]);
              }
              fclose(fp); 
          }
          else
          {
           fclose(fp);
          }  */      
}

template <typename Dtype>
void ManipulatelabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
}

#ifdef CPU_ONLY
STUB_GPU(ManipulatelabelLayer);
#endif

INSTANTIATE_CLASS(ManipulatelabelLayer);
REGISTER_LAYER_CLASS(Manipulatelabel);

}  // namespace caffe
