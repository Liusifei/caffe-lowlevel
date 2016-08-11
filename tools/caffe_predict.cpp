#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp>
#include "time.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;


DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(testfile, "",
    "The test text file..");
DEFINE_string(outputlayername, "",
    "The outputlayername ..");
DEFINE_string(savefolder, "",
    "The savefolder ..");
DEFINE_int32(height, -1,
    "Run in GPU mode on given device ID.");
DEFINE_int32(width, -1,
    "Run in GPU mode on given device ID.");
DEFINE_bool(autoscale, false,
    "Run in GPU mode on given device ID.");
DEFINE_bool(autoresize, false,
    "Run in GPU mode on given device ID.");
DEFINE_string(testimagename, "",
    "The test image file..");

    
    
// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  CHECK_GT(FLAGS_gpu, -1) << "Need a device ID to query.";
  LOG(INFO) << "Querying device ID = " << FLAGS_gpu;
  caffe::Caffe::SetDevice(FLAGS_gpu);
  caffe::Caffe::DeviceQuery();
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}
std::vector<std::string> str_split(std::string str,std::string pattern)
{
    std::string::size_type pos;
    std::vector<std::string> result;
    str+=pattern;
    int size=str.size();

    for(int i=0; i<size; i++)
    {
        pos=str.find(pattern,i);
        if(pos<size)
        {
            std::string s=str.substr(i,pos-i);
            result.push_back(s);
            i=pos+pattern.size()-1;
        }
    }
    return result;
}

void get_caffe_inputdata(cv::Mat img,float * inputdata,bool color=true)
{
  int topindex=0;
  if (color) {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < img.rows; ++h) {
        for (int w = 0; w < img.cols; ++w) {
          float datum_element = static_cast<float>(static_cast<unsigned char>(img.at<cv::Vec3b>(h, w)[c]));
            
           inputdata[topindex++]=(datum_element-128)/255;
            
        }
      }
    }
  } else { 
    for (int h = 0; h < img.rows; ++h) {
      for (int w = 0; w < img.cols; ++w) {
          float datum_element = static_cast<float>(static_cast<char>(img.at<uchar>(h, w)));
           inputdata[topindex++]=(datum_element-128)/255;
        }
      }
  }
}
cv::Mat get_outputmap(const vector<vector<Blob<float>*> >& top_vecs,int outidx,bool auto_scale=false)
{
    
  vector<Blob<float>*> outblobs = top_vecs[outidx];
  const float * outdata=outblobs[0]->cpu_data();
  int count = outblobs[0]->count(); 
  int outheight= outblobs[0]->height();
  int outwidth= outblobs[0]->width();
  int channels = outblobs[0]->channels();
  int spacedim = outheight*outwidth;
  cv::Mat result = cv::Mat(outheight,outwidth,CV_8UC1);

  float maxv=-FLT_MAX;
  int maxid=0;
  float v=0;
  
  int scale_rate=1;
  if(auto_scale)
  {
    scale_rate = 255/(channels-1);
  }
  
  for(int h=0;h<outheight;h++)
  {
    //unsigned char * pdata = result.ptr<unsigned char>(h);
    for(int w=0;w<outwidth;w++)
    {
         
        for(int c=0;c<channels;c++)
        {
          v=outdata[c*spacedim + h* outwidth + w];
          if (v > maxv)
          {
            maxv = v;
            maxid = c;
          }
        }
        if(auto_scale)
        {
            maxid = maxid * scale_rate;
        }
        result.at<unsigned char>(h, w)=(unsigned char)(maxid);
        maxv=-FLT_MAX;
        maxid=0;
    }
  }
  return result;
}
cv::Mat forwardNet(Net<float> &caffe_net,std::string outputlayername, cv::Mat inputimg,int height,int width,bool auto_scale=false,bool auto_resize=false)
{
  int outidx=-1;
  cv::Mat dummyresult;
  int input_height = inputimg.rows;
  int input_width = inputimg.cols;
  cv::Mat cv_img_origin = inputimg;

  cv::Mat cv_img;
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " ;
    return dummyresult;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width,height));
  } else {
    cv_img = cv_img_origin;
  }
    const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
    const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
    const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
    const vector<vector<bool> >& bottom_need_backward = caffe_net.bottom_need_backward();
    
  
  Blob<float>* inputblob = bottom_vecs[0][0];
  float * inputdata = inputblob->mutable_cpu_data();
  get_caffe_inputdata(cv_img,inputdata);
  /////-------------------  fp --------------------------///
  for (int i = 0; i < layers.size(); ++i) {
      const caffe::string& layername = layers[i]->layer_param().name();
      layers[i]->Reshape(bottom_vecs[i], top_vecs[i]);
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      if(outputlayername == layername)
      {
            outidx = i;
            break;
      }
  }
  
  /////-------------------  fp  --------------------------///

  cv::Mat result =  get_outputmap( top_vecs, outidx ,auto_scale);
  if(auto_resize)
  {
    cv::resize(result, result, cv::Size(input_width,input_height),0,0,cv::INTER_NEAREST);
  }
  return result;
}

vector<Blob<float>*> extractNetfeature(Net<float> &caffe_net,std::string outputlayername, cv::Mat inputimg,int height,int width)
{
  int outidx=-1;

  int input_height = inputimg.rows;
  int input_width = inputimg.cols;
  cv::Mat cv_img_origin = inputimg;

  vector<Blob<float>*> dummy;
  cv::Mat cv_img;
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " ;
    return dummy;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width,height));
  } else {
    cv_img = cv_img_origin;
  }
    const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
    const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
    const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
    const vector<vector<bool> >& bottom_need_backward = caffe_net.bottom_need_backward();
    
  
  Blob<float>* inputblob = bottom_vecs[0][0];
  float * inputdata = inputblob->mutable_cpu_data();
  get_caffe_inputdata(cv_img,inputdata);
  if(outputlayername == "data")
  {
    return bottom_vecs[0];
  }
  /////-------------------  fp --------------------------///
  for (int i = 0; i < layers.size(); ++i) {
      const caffe::string& layername = layers[i]->layer_param().name();
      layers[i]->Reshape(bottom_vecs[i], top_vecs[i]);
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      if(outputlayername == layername)
      {
            outidx = i;
            break;
      }
  }
  if(outidx<0)
  {
     LOG(INFO)<<"do not find layer: "<<outputlayername;
     return dummy;
  }
  
  /////-------------------  fp  --------------------------///
  vector<Blob<float>*> outblobs =  top_vecs[outidx];
  return outblobs;
}

int test_saveimg() {

    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
    CHECK_GT(FLAGS_outputlayername.size(), 0) << "Need output layer name.";
    CHECK_GT(FLAGS_savefolder.size(), 0) << "Need save folder.";
    
    bool inputcolorimg=true;
    std::string outputlayername = FLAGS_outputlayername;
    std::string savefolder = FLAGS_savefolder;
    const int height =FLAGS_height;
    const int width = FLAGS_width;
    

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  
    clock_t start, finish;  
    time_t t_start, t_end;   
    
    vector<std::pair<std::string, int> > pair_lines_;
    const std::string source = FLAGS_testfile;
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    std::string filename;
    int label;
    while (infile >> filename >> label) {
        pair_lines_.push_back(std::make_pair(filename, label));
    }
    int length = pair_lines_.size();
    for (int i = 0; i < length; ++i) {
    
        std::vector<std::string> imagenames = str_split(pair_lines_[i].first,"||");
        int cv_read_flag = (inputcolorimg ? CV_LOAD_IMAGE_COLOR :CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat cv_img_origin = cv::imread(imagenames[0], cv_read_flag);
        
        double timecost = (double)cv::getTickCount();
        
        cv::Mat result = forwardNet(caffe_net, outputlayername, cv_img_origin, height, width,FLAGS_autoscale,FLAGS_autoresize);
        
        timecost = ((double)cv::getTickCount() - timecost)*1000/cv::getTickFrequency();
    	if(!result.data)
		{
			LOG(INFO)<<"can not process "<<imagenames[0];
			continue;
		}
        std::vector<std::string> tempname = str_split(imagenames[0],"/");
        std::vector<std::string> tempname2 = str_split(tempname[tempname.size()-1],".");
        std::string savename = tempname2[0];
        savename = savefolder + savename + ".png";
        //LOG(INFO) << i <<"process img " << imagenames[0];
        LOG(INFO) << i << " save img " << savename<<", time="<<timecost<<" ms";
        cv::imwrite(savename, result);
  }
 
  return 0;
}
RegisterBrewFunction(test_saveimg);

int test_extractfeature() {

    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
    CHECK_GT(FLAGS_outputlayername.size(), 0) << "Need output layer name.";
    CHECK_GT(FLAGS_savefolder.size(), 0) << "Need save folder.";
    
    bool inputcolorimg=true;
    std::string outputlayername = FLAGS_outputlayername;
    std::string savefolder = FLAGS_savefolder;
    const int height =FLAGS_height;
    const int width = FLAGS_width;
    

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  
    clock_t start, finish;  
    time_t t_start, t_end;   
    
    vector<std::pair<std::string, int> > pair_lines_;
    const std::string source = FLAGS_testfile;
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    std::string filename;
    int label;
    while (infile >> filename >> label) {
        pair_lines_.push_back(std::make_pair(filename, label));
    }
    int length = pair_lines_.size();
    for (int i = 0; i < length; ++i) {
    
        std::vector<std::string> imagenames = str_split(pair_lines_[i].first,"||");
        int cv_read_flag = (inputcolorimg ? CV_LOAD_IMAGE_COLOR :CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat cv_img_origin = cv::imread(imagenames[0], cv_read_flag);
        
        double timecost = (double)cv::getTickCount();
        
        vector<Blob<float>*> outblobs = extractNetfeature(caffe_net, outputlayername, cv_img_origin, height, width);
        
        timecost = ((double)cv::getTickCount() - timecost)*1000/cv::getTickFrequency();
    	if(outblobs.size()<1||!outblobs[0]->count())
		{
			LOG(INFO)<<"can not process "<<imagenames[0];
			continue;
		}
        std::vector<std::string> tempname = str_split(imagenames[0],"/");
        std::vector<std::string> tempname2 = str_split(tempname[tempname.size()-1],".");
        std::string savename = tempname2[0];
        savename = savefolder + savename + ".txt";
        //LOG(INFO) << i <<"process img " << imagenames[0];
        LOG(INFO) << i << " save data " << savename<<", time="<<timecost<<" ms";
        
        const float * outdata=outblobs[0]->cpu_data();
        FILE *fp =fopen(savename.c_str(),"w");
        for(int num=0;num<outblobs[0]->count();num++)
        {
            fprintf(fp,"%f\n",outdata[num]);
        }
        fclose(fp);
        //cv::imwrite(savename, result);
  }
 
  return 0;
}
RegisterBrewFunction(test_extractfeature);

int test_extract_one_feature() {

    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
    CHECK_GT(FLAGS_outputlayername.size(), 0) << "Need output layer name.";
    CHECK_GT(FLAGS_savefolder.size(), 0) << "Need save folder.";
    
    bool inputcolorimg=true;
    std::string outputlayername = FLAGS_outputlayername;
    std::string savefolder = FLAGS_savefolder;
    const int height =FLAGS_height;
    const int width = FLAGS_width;
    

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  
    clock_t start, finish;  
    time_t t_start, t_end;   
    
 
    const std::string imagename = FLAGS_testimagename;
    LOG(INFO) << "Opening file " << imagename;
    
    
       int cv_read_flag = (inputcolorimg ? CV_LOAD_IMAGE_COLOR :CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat cv_img_origin = cv::imread(imagename, cv_read_flag);
        
        double timecost = (double)cv::getTickCount();
        
        vector<Blob<float>*> outblobs = extractNetfeature(caffe_net, outputlayername, cv_img_origin, height, width);
        
        timecost = ((double)cv::getTickCount() - timecost)*1000/cv::getTickFrequency();
    	if(outblobs.size()<1||!outblobs[0]->count())
		{
			LOG(INFO)<<"can not process "<<imagename;
			return 0;
		}
        std::vector<std::string> tempname = str_split(imagename,"/");
        std::vector<std::string> tempname2 = str_split(tempname[tempname.size()-1],".");
        std::string savename = tempname2[0];
        savename = savefolder + savename + ".txt";
        //LOG(INFO) << i <<"process img " << imagenames[0];
        LOG(INFO)<< " save data " << savename<<", time="<<timecost<<" ms";
        
        const float * outdata=outblobs[0]->cpu_data();
        FILE *fp =fopen(savename.c_str(),"w");
        for(int num=0;num<outblobs[0]->count();num++)
        {
            fprintf(fp,"%f\n",outdata[num]);
        }
        fclose(fp);
        //cv::imwrite(savename, result);
  
 
  return 0;
}
RegisterBrewFunction(test_extract_one_feature);

int test_extractparam() {

    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
    CHECK_GT(FLAGS_outputlayername.size(), 0) << "Need output layer name.";
    CHECK_GT(FLAGS_savefolder.size(), 0) << "Need save folder.";
    
    bool inputcolorimg=true;
    std::string outputlayername = FLAGS_outputlayername;
    std::string savefolder = FLAGS_savefolder;
    const int height =FLAGS_height;
    const int width = FLAGS_width;
    

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  
  if(outputlayername == "ALL")
  {
     vector<std::string> layernames =  caffe_net.layer_names();
     for(int k=0;k<layernames.size();k++)
     {
        shared_ptr<Layer<float> > layer =  caffe_net.layer_by_name(layernames[k]);
        vector<shared_ptr<Blob<float> > > paramblobs =  layer->blobs();
        LOG(INFO)<<layernames[k]<<" have "<<paramblobs.size()<<" params.";
        for(int i=0;i<paramblobs.size();i++)
        {
            std::stringstream ss;
            std::string str;
            ss<<i;
            ss>>str;
            std::string savename = layernames[k]+"_param_"+str+".txt";
            const float * paramdata=paramblobs[i]->cpu_data();
            FILE *fp =fopen(savename.c_str(),"w");
            for(int num=0;num<paramblobs[i]->count();num++)
            {
                fprintf(fp,"%f\n",paramdata[num]);
            }
            fclose(fp);
        }
     }
  }
  else
  {
    shared_ptr<Layer<float> > layer =  caffe_net.layer_by_name(outputlayername);
    vector<shared_ptr<Blob<float> > > paramblobs =  layer->blobs();
    LOG(INFO)<<outputlayername<<" have "<<paramblobs.size()<<" params.";
    for(int i=0;i<paramblobs.size();i++)
    {
        std::stringstream ss;
        std::string str;
        ss<<i;
        ss>>str;
        std::string savename = outputlayername+"_param_"+str+".txt";
        const float * paramdata=paramblobs[i]->cpu_data();
        FILE *fp =fopen(savename.c_str(),"w");
        for(int num=0;num<paramblobs[i]->count();num++)
        {
            fprintf(fp,"%f\n",paramdata[num]);
        }
        fclose(fp);
    }
  }


    
   
 
  return 0;
}
RegisterBrewFunction(test_extractparam);

int main(int argc, char** argv) {
    // Print output to stderr (while still logging).
    FLAGS_alsologtostderr = 1;
    // Usage message.
    gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  test_saveimg    test and save result as image\n"
      "  test_extractfeature  extract and save feature data \n");
    // Run tool or show usage.
    caffe::GlobalInit(&argc, &argv);
    if (argc == 2) {
    return GetBrewFunction(caffe::string(argv[1]))();
    } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
    }
}
