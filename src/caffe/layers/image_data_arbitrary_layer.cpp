#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
    
    template <typename Dtype>
        ImageDataArbitraryLayer<Dtype>::~ImageDataArbitraryLayer<Dtype>() {
            this->JoinPrefetchThread();
            delete [] datamean;
            delete [] dummyLabelmean;
        }

    template <typename Dtype>
        void ImageDataArbitraryLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {
		
            this->top_size_ = top.size();
			//CHECK(this->top_size_ == top->size())<<"top size error! "<<top->size()
            LOG(INFO) << "top.size()="<<top.size();
            this->prefetch_data_.clear();
            LOG(INFO) << "this->prefetch_data_ size="<<this->prefetch_data_.size();
			for(int i=0;i<this->top_size_;i++)
			  {
				shared_ptr<Blob <Dtype> > blob_pointer(new Blob<Dtype>());
				this->prefetch_data_.push_back(blob_pointer);
			  }
            LOG(INFO) << "this->prefetch_data_ size after pushback="<<this->prefetch_data_.size();
  
  
			//LOG(INFO) << "this->prefetch_data_.size()  "<<this->prefetch_data_.size(); 
            srand( time((unsigned)NULL) );


            this->data_height_ = this->layer_param_.image_data_arbitrary_param().data_height();
            this->data_width_  = this->layer_param_.image_data_arbitrary_param().data_width();
			this->data_channels_  = this->layer_param_.image_data_arbitrary_param().data_channels();

            //add by liangji
            this->label_height_ = this->layer_param_.image_data_arbitrary_param().label_height();
            this->label_width_  = this->layer_param_.image_data_arbitrary_param().label_width();
			this->batch_size_ = this->layer_param_.image_data_arbitrary_param().batch_size();
			this->data_scale_ = this->layer_param_.image_data_arbitrary_param().data_scale();

            string meanfile = "";
            if (this->layer_param_.image_data_arbitrary_param().has_meanfile())
            {
                meanfile = this->layer_param_.image_data_arbitrary_param().meanfile();
            }


            datamean=NULL;
            dummyLabelmean=NULL;
            int datasize = this->data_height_*this->data_width_*this->data_channels_;
            int labelsize = this->label_height_ * this->label_width_;
            datamean=new Dtype [datasize];
            if(datamean == NULL)
            {
                CHECK(true) << "can not create datamean in ImageDataArbitraryLayer";
            }
            if(meanfile.size()<1)
            {
                for(int tempidx=0;tempidx<datasize;tempidx++)
                {
                    datamean[tempidx] = 128.0;
                }
            }
            else
            {
                printf("load data mean from %s\n",meanfile.c_str());
                FILE * fp =fopen(meanfile.c_str(),"r");
                    int k=0;
                    for( k=0;k<datasize;k++) 
                    {
                        fscanf(fp,"%f\n",&datamean[k]);
                    }
                fclose(fp);
            }


            dummyLabelmean=new Dtype [labelsize];
            if(dummyLabelmean == NULL)
            {
                CHECK(true) << "can not create dummyLabelmean in ImageDataArbitraryLayer";
            }
            for(int tempidx=0;tempidx<labelsize;tempidx++)
            {
                dummyLabelmean[tempidx] = 0;
            }

            CHECK((data_height_ == 0 && data_width_ == 0) ||
                    (data_height_ > 0 && data_width_ > 0)) << "Current implementation requires "
                "data_height and data_width to be set at the same time.";

            CHECK( (label_height_ > 0 && label_width_ > 0)) << "Current implementation requires "
                "label_height and label_width must to be set.";


            // Read the file with filenames and labels
            const string& source = this->layer_param_.image_data_arbitrary_param().source();
            LOG(INFO) << "Opening file " << source;
            std::ifstream infile(source.c_str());
            string filename;
            int label;
            while (infile >> filename >> label) {
                pair_lines_.push_back(std::make_pair(filename, label));
            }

            if (this->layer_param_.image_data_arbitrary_param().shuffle()) {
                // randomly shuffle data
                LOG(INFO) << "Shuffling data";
                const unsigned int prefetch_rng_seed = caffe_rng_rand();
                prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
                ShuffleImages();
            }
            LOG(INFO) << "A total of " << pair_lines_.size() << " images.";

            lines_id_ = 0;
            // Check if we would need to randomly skip a few data points
            if (this->layer_param_.image_data_arbitrary_param().rand_skip()) {
                unsigned int skip = caffe_rng_rand() %
                    this->layer_param_.image_data_arbitrary_param().rand_skip();
                LOG(INFO) << "Skipping first " << skip << " data points.";
                CHECK_GT(pair_lines_.size(), skip) << "Not enough points to skip";
                lines_id_ = skip;
            }
           
            LOG(INFO) << "before reshape data"; 
         
			top[0]->Reshape(this->batch_size_, this->data_channels_, this->data_height_,
					this->data_width_);
			LOG(INFO) << "after reshape top"; 
			LOG(INFO) << "prefetch_data_ size"<<this->prefetch_data_.size(); 
			this->prefetch_data_[0]->Reshape(this->batch_size_,this->data_channels_, this->data_height_,
					this->data_width_);
			LOG(INFO) << "in the mid reshape data"; 
			for(int i=1;i<this->top_size_;i++)
			{
				top[i]->Reshape(this->batch_size_, 1, this->label_height_,
					this->label_width_);
				this->prefetch_data_[i]->Reshape(this->batch_size_,1, this->label_height_,
					this->label_width_);
			}
            LOG(INFO) << "after reshape data"; 
            
            LOG(INFO) << "output data size: " << top[0]->num() << ","
                << top[0]->channels() << "," << top[0]->height() << ","
                << top[0]->width();
				
			//just for BaseDataLayer check, do not used
            /*
			this->datum_channels_ = this->data_channels_;
            this->datum_height_ = this->data_height_;
            this->datum_width_ = this->data_width_;
            this->datum_size_ = this->data_channels_ * this->data_height_ * this->data_width_;
            */
        }

    template <typename Dtype>
        void ImageDataArbitraryLayer<Dtype>::ShuffleImages() {
            caffe::rng_t* prefetch_rng =
                static_cast<caffe::rng_t*>(prefetch_rng_->generator());
            shuffle(pair_lines_.begin(), pair_lines_.end(), prefetch_rng);
        }

    // This function is used to create a thread that prefetches the data.
    template <typename Dtype>
        void ImageDataArbitraryLayer<Dtype>::transData(const int batch_item_id,const Datum& datum,const Dtype* mean,Dtype* transformed_data,const Dtype scale)
        {
            const string& data = datum.data();
            //const int channels = datum.channels();
            //const int height = datum.height();
            //const int width = datum.width();
            const int size = datum.channels() * datum.height() * datum.width();

            if (data.size()) 
            {
                for (int j = 0; j < size; ++j) 
                {
                    Dtype datum_element =static_cast<Dtype>(static_cast<uint8_t>(data[j]));
                    transformed_data[j + batch_item_id * size] = (datum_element - mean[j]) * scale;
                }
            }
            else 
            {
                for (int j = 0; j < size; ++j) 
                {
                    transformed_data[j + batch_item_id * size] = (datum.float_data(j) - mean[j]) * scale;
                }
            }
        }

    // This function is used to create a thread that prefetches the data.
    template <typename Dtype>
        void ImageDataArbitraryLayer<Dtype>::InternalThreadEntry() {
            
           // LOG(INFO) << "in ImageDataArbitraryLayer<Dtype>::InternalThreadEntry";
            CHECK(this->prefetch_data_[0]->count());
            //Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
            
            ImageDataArbitraryParameter image_data_arbitrary_param = this->layer_param_.image_data_arbitrary_param();
            
			vector< int> channelNum;
			vector< int> height;
			vector < int> width;
			vector< Datum > datum;
			vector<int> resizetype;
			
			channelNum.push_back(this->data_channels_);
			height.push_back(this->data_height_);
			width.push_back(this->data_width_);
			resizetype.push_back(0);
			Datum onedatum;
			datum.push_back(onedatum);
			for(int i=1;i< this->top_size_;i++)
			{
				channelNum.push_back(1);
				height.push_back(this->label_height_);
				width.push_back(this->label_width_);
				resizetype.push_back(1);
				Datum onedatum;
				datum.push_back(onedatum);
			}
			
			

            // datum scales
            const int lines_size = pair_lines_.size();
            for (int item_id = 0; item_id < this->batch_size_; ++item_id) {
                //std::string imagefilename = pair_lines_[lines_id_].first;
                //std::int labelname = pair_lines_[lines_id_].second;


                
                float angle,scale,move_positoin_x,move_positoin_y;
                bool flipimg=false;
                bool is_add_grid_perturb = false;
				bool useflip = image_data_arbitrary_param.useflip();
				bool usemovedisturb = image_data_arbitrary_param.usemovedisturb();
                if(image_data_arbitrary_param.use_disturb())
                {
                    //printf("in train\n");
                    angle=30* ((caffe::caffe_rng_rand()%INT_MAX)*1.0f / (INT_MAX-1) - 0.5); //rotation in -15~15 degree
                    scale = 0.3* ((caffe::caffe_rng_rand()%INT_MAX)*1.0f / (INT_MAX-1)  - 0.5) +1; // scale in 0.85~1.15
                    move_positoin_x= 0.3*((caffe::caffe_rng_rand()%INT_MAX)*1.0f / (INT_MAX-1) - 0.5); //transform (-0.15~0.15) of new width
                    move_positoin_y= 0.3*((caffe::caffe_rng_rand()%INT_MAX)*1.0f / (INT_MAX-1) - 0.5); //transform (-0.15~0.15) of new height
                    flipimg=((caffe::caffe_rng_rand()%INT_MAX)*1.0f / (INT_MAX-1)) > 0.5 ? true: false;
                    is_add_grid_perturb = ((caffe::caffe_rng_rand()%INT_MAX)*1.0f / (INT_MAX-1)) > 0.6 ? true: false;
                    //std::cerr<<"augment param: "<<angle<<" "<<scale<<" "<<move_positoin_x<<" "<<move_positoin_y<<" "<<flipimg<<" "<<is_add_grid_perturb<<std::endl;
                }
                else
                {
                    //printf("in test\n");
                    angle = 0;
                    scale = 1;
                    move_positoin_x=0;
                    move_positoin_y =0;
                    flipimg = false;
                    is_add_grid_perturb = false;

                }
				
				if(!useflip)
				{
					flipimg = false;
				}
				if(!usemovedisturb)
				{
					move_positoin_x=0;
                    move_positoin_y =0;
				}
				
				
                // get a blob
                CHECK_GT(lines_size, lines_id_);
				
				//LOG(INFO) << " scale:"<<scale<<" angle:"<<angle<<" move_positoin_x:"<<move_positoin_x<<" move_positoin_y:"<<move_positoin_y<<" flipimg:"<<flipimg<<" is_add_grid_perturb:"<<is_add_grid_perturb;

				if (!Read_Images_ToDatum(pair_lines_[lines_id_].first,channelNum,
                            height, width,datum,angle,scale,is_add_grid_perturb,move_positoin_x,move_positoin_y,flipimg,resizetype)) {
                    continue;
                }
				
				this->transData(item_id, datum[0], this->datamean, this->prefetch_data_[0]->mutable_cpu_data(),this->data_scale_);
				for(int i=1;i<this->top_size_;i++)
				{
					this->transData(item_id, datum[i],this->dummyLabelmean, this->prefetch_data_[i]->mutable_cpu_data(),1);
				}
                
                // go to the next iter
                lines_id_++;
                if (lines_id_ >= lines_size) {
                    // We have reached the end. Restart from the first.
                    DLOG(INFO) << "Restarting data prefetching from start.";
                    lines_id_ = 0;
                    if (this->layer_param_.image_data_param().shuffle()) {
                        ShuffleImages();
                    }
                }
            }
        }

    
    INSTANTIATE_CLASS(ImageDataArbitraryLayer);
    REGISTER_LAYER_CLASS(ImageDataArbitrary);

}  // namespace caffe
