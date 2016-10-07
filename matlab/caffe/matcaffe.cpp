//
// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.

#include <sstream>
#include <string>
#include <vector>

#include "mex.h"

#include "caffe/caffe.hpp"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

// Log and throw a Mex error
inline void mex_error(const std::string &msg) {
  LOG(ERROR) << msg;
  mexErrMsgTxt(msg.c_str());
}

using namespace caffe;  // NOLINT(build/namespaces)

// The pointer to the internal caffe::Net instance
static shared_ptr<Solver<float> > solver_;
static shared_ptr<Net<float> > net_;
static shared_ptr<Net<float> > test_net_;
static int init_key = -2;

// Five things to be aware of:
//   caffe uses row-major order
//   matlab uses column-major order
//   caffe uses BGR color channel order
//   matlab uses RGB color channel order
//   images need to have the data mean subtracted
//
// Data coming in from matlab needs to be in the order
//   [width, height, channels, images]
// where width is the fastest dimension.
// Here is the rough matlab for putting image data into the correct
// format:
//   % convert from uint8 to single
//   im = single(im);
//   % reshape to a fixed size (e.g., 227x227)
//   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
//   % permute from RGB to BGR and subtract the data mean (already in BGR)
//   im = im(:,:,[3 2 1]) - data_mean;
//   % flip width and height to make width the fastest dimension
//   im = permute(im, [2 1 3]);
//
// If you have multiple images, cat them with cat(4, ...)
//
// The actual forward function. It takes in a cell array of 4-D arrays as
// input and outputs a cell array.

static mxArray* do_forward(const mxArray* const bottom) {
  const vector<Blob<float>*>& input_blobs = net_->input_blobs();
  if (static_cast<unsigned int>(mxGetDimensions(bottom)[0]) !=
      input_blobs.size()) {
    mex_error("Invalid input size");
  }
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
    if (!mxIsSingle(elem)) {
      mex_error("MatCaffe require single-precision float point data");
    }
    if (mxGetNumberOfElements(elem) != input_blobs[i]->count()) {
      std::string error_msg;
      error_msg += "MatCaffe input size does not match the input size ";
      error_msg += "of the network";
      mex_error(error_msg);
    }

    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_cpu_data());
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_gpu_data());
      break;
    default:
      mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }
  const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
  mxArray* mx_out = mxCreateCellMatrix(output_blobs.size(), 1);
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {output_blobs[i]->width(), output_blobs[i]->height(),
      output_blobs[i]->channels(), output_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->cpu_data(),
          data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->gpu_data(),
          data_ptr);
      break;
    default:
      mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

static mxArray* do_forward_test(const mxArray* const bottom, 
const mxArray* const shape) {
  const vector<Blob<float>*>& input_blobs = test_net_->input_blobs();
  //double* input_shape = (double*)mxGetPr(shape);
  if (static_cast<unsigned int>(mxGetDimensions(bottom)[0]) !=
      input_blobs.size()) {
    mex_error("Invalid input size");
  }
   double input_shape = static_cast<double>(mxGetScalar(shape));
   LOG(INFO) << "reshaping input blobs.";
   for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i); 
    if (!mxIsSingle(elem)) {
      mex_error("MatCaffe require single-precision float point data");
    }   
    if (mxGetNumberOfElements(elem) != input_blobs[i]->count()) {
      //std::string error_msg;
      //error_msg += "MatCaffe input size does not match the input size ";
      //error_msg += "of the network";
      //mex_error(error_msg);
      //shared_ptr<Blob<float> > blob_pointer(new Blob<float>());
      //LOG(INFO) << "reshape input blobs.";
      input_blobs[i]->Reshape(1,5,input_shape,input_shape);
      LOG(INFO) << "input blobs reshaped to input image size.";
    }   

    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_cpu_data());
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_gpu_data());
      break;
    default:
      mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }
  const vector<Blob<float>*>& output_blobs = test_net_->ForwardPrefilled();
  mxArray* mx_out = mxCreateCellMatrix(output_blobs.size(), 1); 
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {output_blobs[i]->width(), output_blobs[i]->height(),
      output_blobs[i]->channels(), output_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->cpu_data(),
          data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->gpu_data(),
          data_ptr);
      break;
    default:
      mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

static mxArray* do_backward(const mxArray* const top_diff) {
  const vector<Blob<float>*>& output_blobs = net_->output_blobs();
  const vector<Blob<float>*>& input_blobs = net_->input_blobs();
  if (static_cast<unsigned int>(mxGetDimensions(top_diff)[0]) !=
      output_blobs.size()) {
    mex_error("Invalid input size");
  }
  // First, copy the output diff
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(top_diff, i);
    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_cpu_diff());
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_gpu_diff());
      break;
    default:
        mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }
  // LOG(INFO) << "Start";
  net_->Backward();
  // LOG(INFO) << "End";
  mxArray* mx_out = mxCreateCellMatrix(input_blobs.size(), 1);
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {input_blobs[i]->width(), input_blobs[i]->height(),
      input_blobs[i]->channels(), input_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->cpu_diff(), data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->gpu_diff(), data_ptr);
      break;
    default:
        mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

static mxArray* do_get_weights() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

  // Step 1: count the number of layers with weights
  int num_layers = 0;
  {
    string prev_layer_name = "";
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        num_layers++;
      }
    }
  }

  // Step 2: prepare output array of structures
  mxArray* mx_layers;
  {
    const mwSize dims[2] = {num_layers, 1};
    const char* fnames[2] = {"weights", "layer_names"};
    mx_layers = mxCreateStructArray(2, dims, 2, fnames);
  }

  // Step 3: copy weights into output
  {
    string prev_layer_name = "";
    int mx_layer_index = 0;
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }

      mxArray* mx_layer_cells = NULL;
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        const mwSize dims[2] = {static_cast<mwSize>(layer_blobs.size()), 1};
        mx_layer_cells = mxCreateCellArray(2, dims);
        mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_cells);
        mxSetField(mx_layers, mx_layer_index, "layer_names",
            mxCreateString(layer_names[i].c_str()));
        mx_layer_index++;
      }

      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
            layer_blobs[j]->channels(), layer_blobs[j]->num()};

        mxArray* mx_weights =
          mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
        mxSetCell(mx_layer_cells, j, mx_weights);
        float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

        switch (Caffe::mode()) {
        case Caffe::CPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->cpu_data(),
              weights_ptr);
          break;
        case Caffe::GPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->gpu_data(),
              weights_ptr);
          break;
        default:
          mex_error("Unknown Caffe mode");
        }
      }
    }
  }

  return mx_layers;
}

static mxArray* do_get_weights_test() {
  const vector<shared_ptr<Layer<float> > >& layers = test_net_->layers();
  const vector<string>& layer_names = test_net_->layer_names();
  LOG(INFO)<<"set test net done";
  // Step 1: count the number of layers with weights
  int num_layers = 0;
  {
    string prev_layer_name = "";
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        num_layers++;
      }
    }
  }

  // Step 2: prepare output array of structures
  mxArray* mx_layers;
  {
    const mwSize dims[2] = {num_layers, 1};
    const char* fnames[2] = {"weights", "layer_names"};
    mx_layers = mxCreateStructArray(2, dims, 2, fnames);
  }

  // Step 3: copy weights into output
  {
    string prev_layer_name = "";
    int mx_layer_index = 0;
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }

      mxArray* mx_layer_cells = NULL;
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        const mwSize dims[2] = {static_cast<mwSize>(layer_blobs.size()), 1};
        mx_layer_cells = mxCreateCellArray(2, dims);
        mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_cells);
        mxSetField(mx_layers, mx_layer_index, "layer_names",
            mxCreateString(layer_names[i].c_str()));
        mx_layer_index++;
      }

      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
            layer_blobs[j]->channels(), layer_blobs[j]->num()};

        mxArray* mx_weights =
          mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
        mxSetCell(mx_layer_cells, j, mx_weights);
        float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

        switch (Caffe::mode()) {
        case Caffe::CPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->cpu_data(),
              weights_ptr);
          break;
        case Caffe::GPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->gpu_data(),
              weights_ptr);
          break;
        default:
          mex_error("Unknown Caffe mode");
        }
      }
    }
  }

  return mx_layers;
}

static void do_set_layer_weights(const mxArray* const layer_name,
    const mxArray* const mx_layer_weights) {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  //LOG(INFO) << "Done with get solver layers.";
  const vector<string>& layer_names = net_->layer_names();
  //LOG(INFO) << "Done with get solver layer names.";

  char* c_layer_names = mxArrayToString(layer_name);
  LOG(INFO) << "Looking for: " << c_layer_names;

  for (unsigned int i = 0; i < layers.size(); ++i) {
    DLOG(INFO) << layer_names[i];
    if (strcmp(layer_names[i].c_str(),c_layer_names) == 0) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      DLOG(INFO) << "Found layer " << layer_names[i] << "layer_blobs.size() = " << layer_blobs.size();
      if (static_cast<unsigned int>(mxGetDimensions(mx_layer_weights)[0]) != layer_blobs.size()) {
            mex_error("Num of cells don't match layer_blobs.size");
      }
      LOG(INFO) << "layer_blobs.size() = " << layer_blobs.size();
      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        const mxArray* const elem = mxGetCell(mx_layer_weights, j);
        mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
            layer_blobs[j]->channels(), layer_blobs[j]->num()};
        DLOG(INFO) << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3];
        if (layer_blobs[j]->count() != mxGetNumberOfElements(elem)) {
            mex_error("Numel of weights don't match count of layer_blob");
        }
        const mwSize* dims_elem = mxGetDimensions(elem);
        DLOG(INFO) << dims_elem[0] << " " << dims_elem[1];
        const float* const data_ptr =
            reinterpret_cast<const float* const>(mxGetPr(elem));
        DLOG(INFO) << "elem: " << data_ptr[0] << " " << data_ptr[1];
        DLOG(INFO) << "count: " << layer_blobs[j]->count();
        switch (Caffe::mode()) {
        case Caffe::CPU:
          memcpy(layer_blobs[j]->mutable_cpu_data(), data_ptr,
              sizeof(float) * layer_blobs[j]->count());
          break;
        case Caffe::GPU:
          cudaMemcpy(layer_blobs[j]->mutable_gpu_data(), data_ptr,
              sizeof(float) * layer_blobs[j]->count(), cudaMemcpyHostToDevice);
          break;
        default:
          LOG(FATAL) << "Unknown Caffe mode.";
        }
      }
    }
  }
}


static void do_set_layer_weights_test(const mxArray* const layer_name,
  const mxArray* const mx_layer_weights) {
  const vector<shared_ptr<Layer<float> > >& layers = test_net_->layers();
  //LOG(INFO) << "Done with get solver layers.";
  const vector<string>& layer_names = test_net_->layer_names();
  //LOG(INFO) << "Done with get solver layer names.";

  char* c_layer_names = mxArrayToString(layer_name);
  LOG(INFO) << "Looking for: " << c_layer_names;

  for (unsigned int i = 0; i < layers.size(); ++i) {
    DLOG(INFO) << layer_names[i];
    if (strcmp(layer_names[i].c_str(),c_layer_names) == 0) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }   
      DLOG(INFO) << "Found layer " << layer_names[i] << "layer_blobs.size() = " << layer_blobs.size();
      if (static_cast<unsigned int>(mxGetDimensions(mx_layer_weights)[0]) != layer_blobs.size()) {
            mex_error("Num of cells don't match layer_blobs.size");
      }   
      LOG(INFO) << "layer_blobs.size() = " << layer_blobs.size();
      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        const mxArray* const elem = mxGetCell(mx_layer_weights, j); 
        mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
            layer_blobs[j]->channels(), layer_blobs[j]->num()};
        DLOG(INFO) << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3];
        if (layer_blobs[j]->count() != mxGetNumberOfElements(elem)) {
            mex_error("Numel of weights don't match count of layer_blob");
        }   
        const mwSize* dims_elem = mxGetDimensions(elem);
        DLOG(INFO) << dims_elem[0] << " " << dims_elem[1];
        const float* const data_ptr =
            reinterpret_cast<const float* const>(mxGetPr(elem));
        DLOG(INFO) << "elem: " << data_ptr[0] << " " << data_ptr[1];
        DLOG(INFO) << "count: " << layer_blobs[j]->count();
        switch (Caffe::mode()) {
        case Caffe::CPU:
          memcpy(layer_blobs[j]->mutable_cpu_data(), data_ptr,
              sizeof(float) * layer_blobs[j]->count());
          break;
        case Caffe::GPU:
          cudaMemcpy(layer_blobs[j]->mutable_gpu_data(), data_ptr,
              sizeof(float) * layer_blobs[j]->count(), cudaMemcpyHostToDevice);
          break;
        default:
          LOG(FATAL) << "Unknown Caffe mode.";
        }   
      }   
    }   
  }
}

static mxArray* do_get_all_data() {
  const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();
  const vector<string>& blob_names = net_->blob_names();

  // Step 1: prepare output array of structures
  mxArray* mx_all_data;
  {
    const int num_blobs[1] = {blobs.size()};
    const char* fnames[2] = {"name", "data"};
    mx_all_data = mxCreateStructArray(1, num_blobs, 2, fnames);
  }

  for (unsigned int i = 0; i < blobs.size(); ++i) {
    DLOG(INFO) << blob_names[i];
    mwSize dims[4] = {blobs[i]->width(), blobs[i]->height(),
        blobs[i]->channels(), blobs[i]->num()};
    DLOG(INFO) << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3];
    mxArray* mx_blob_data =
      mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);

    float* blob_data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob_data));

    switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(blob_data_ptr, blobs[i]->cpu_data(),
          sizeof(float) * blobs[i]->count());
      break;
    case Caffe::GPU:
      CUDA_CHECK(cudaMemcpy(blob_data_ptr, blobs[i]->gpu_data(),
          sizeof(float) * blobs[i]->count(), cudaMemcpyDeviceToHost));
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
    mxSetField(mx_all_data, i, "name",
        mxCreateString(blob_names[i].c_str()));
    mxSetField(mx_all_data, i, "data",mx_blob_data);
  }
  return mx_all_data;
}


static mxArray* do_get_all_diff() {
  const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();
  const vector<string>& blob_names = net_->blob_names();

  // Step 1: prepare output array of structures
  mxArray* mx_all_diff;
  {
    const int num_blobs[1] = {blobs.size()};
    const char* fnames[2] = {"name", "diff"};
    mx_all_diff = mxCreateStructArray(1, num_blobs, 2, fnames);
  }

  for (unsigned int i = 0; i < blobs.size(); ++i) {
    DLOG(INFO) << blob_names[i];
    mwSize dims[4] = {blobs[i]->width(), blobs[i]->height(),
        blobs[i]->channels(), blobs[i]->num()};
    DLOG(INFO) << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3];
    mxArray* mx_blob_data =
      mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);

    float* blob_data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob_data));

    switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(blob_data_ptr, blobs[i]->cpu_diff(),
          sizeof(float) * blobs[i]->count());
      break;
    case Caffe::GPU:
      CUDA_CHECK(cudaMemcpy(blob_data_ptr, blobs[i]->gpu_diff(),
          sizeof(float) * blobs[i]->count(), cudaMemcpyDeviceToHost));
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
    mxSetField(mx_all_diff, i, "name",
        mxCreateString(blob_names[i].c_str()));
    mxSetField(mx_all_diff, i, "diff",mx_blob_data);
  }
  return mx_all_diff;
}

static void get_weights(MEX_ARGS) {
 if (nrhs != 1) {
     LOG(ERROR) << "Given " << nrhs << " arguments expecting 1";
     mexErrMsgTxt("Wrong number of arguments");
  }
  char* phase_name = mxArrayToString(prhs[0]);
  if (strcmp(phase_name, "train") == 0) {
      if (!net_) {
         LOG(ERROR) << "train net is not init successfully";
         mexErrMsgTxt("train net is not init successfully");
      }
      LOG(INFO)<<"get weights train";
      plhs[0] = do_get_weights();
  } else if (strcmp(phase_name, "test") == 0) {
      if (!test_net_) {
         LOG(ERROR) << "test net is not init successfully";
         mexErrMsgTxt("test net is not init successfully");
      }
      LOG(INFO)<<"get weights test";
      plhs[0] = do_get_weights_test();
  }
}

static void set_weights(MEX_ARGS) {
 if (nrhs != 2) {
     LOG(ERROR) << "Given " << nrhs << " arguments expecting 2";
     mexErrMsgTxt("Wrong number of arguments");
  }
  const mxArray* const mx_weights = prhs[0];
  char* phase_name = mxArrayToString(prhs[1]);
  if (!mxIsStruct(mx_weights)) {
     mexErrMsgTxt("Input needs to be struct");
  }
  int num_layers = mxGetNumberOfElements(mx_weights);
  // LOG(INFO) << "begin set layers with layer number: " << num_layers;
  for (int i = 0; i < num_layers; ++i) {
    const mxArray* layer_name= mxGetField(mx_weights,i,"layer_names");
    const mxArray* weights= mxGetField(mx_weights,i,"weights");
    if (strcmp(phase_name, "train") == 0) {
        do_set_layer_weights(layer_name,weights);
    } else if (strcmp(phase_name, "test") == 0) {
        do_set_layer_weights_test(layer_name,weights);
    }
  }
}


static void get_all_data(MEX_ARGS) {
  if (nrhs != 0) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  plhs[0] = do_get_all_data();
}

static void get_all_diff(MEX_ARGS) {
  if (nrhs != 0) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  plhs[0] = do_get_all_diff();
}


static void set_mode_cpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::CPU);
}

static void set_mode_gpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::GPU);
}

static void set_device(MEX_ARGS) {
  if (nrhs != 1) {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id);
}

static void get_init_key(MEX_ARGS) {
  plhs[0] = mxCreateDoubleScalar(init_key);
}

static void init(MEX_ARGS) {
  if (nrhs != 2) {
    ostringstream error_msg;
    error_msg << "Expected 2 arguments, got " << nrhs;
    mex_error(error_msg.str());
  }

  //char* param_file = mxArrayToString(prhs[0]);
  //char* model_file = mxArrayToString(prhs[1]);
  char* solver_file = mxArrayToString(prhs[0]);
  char* phase_name = mxArrayToString(prhs[1]);

  Phase phase;
  if (strcmp(phase_name, "train") == 0) {
      SolverParameter solver_param;
      ReadProtoFromTextFileOrDie(solver_file, &solver_param);
      solver_.reset(caffe::GetSolver<float>(solver_param));
      net_ = solver_->net();
  } else if (strcmp(phase_name, "test") == 0) {
      phase = TEST;
      test_net_.reset(new Net<float>(string(solver_file), phase));
  } else {
    mex_error("Unknown phase.");
  }
    
  //solver_.reset(new SGDSolver<float>(solver_file));
  //net_.reset(new Net<float>(string(param_file), phase));
  //net_->CopyTrainedLayersFrom(string(model_file));
  //mxFree(param_file);
  //mxFree(model_file);
  mxFree(solver_file);
  mxFree(phase_name);

  init_key = random();  // NOLINT(caffe/random_fn)

  if (nlhs == 1) {
    plhs[0] = mxCreateDoubleScalar(init_key);
  }
}

static void reset(MEX_ARGS) {
  if (net_) {
    net_.reset();
    init_key = -2;
    LOG(INFO) << "Network reset, call init before use it again";
  }
  if (test_net_){
    test_net_.reset();
    init_key = -2;
    LOG(INFO) << "Network reset, call init before use it again";
  }
}

//static void presolve(MEX_ARGS){
//  if (solver_) {
 //       solver_->PreSolve();
  //      LOG(INFO) << "Presolve done...";
  //}
//}

static void forward(MEX_ARGS) {
  if (nrhs != 1) {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  plhs[0] = do_forward(prhs[0]);
}

static void forward_test(MEX_ARGS) {
  if (nrhs != 2) {
    ostringstream error_msg;
    error_msg << "Expected 2 argument, got " << nrhs;
    mex_error(error_msg.str());
  }
  //const mxArray* const input_shape = mxGetCell(prhs[1], 1);
  //shape = static_cast<int>(mxGetScalar(prhs[1]))
  plhs[0] = do_forward_test(prhs[0], prhs[1]);
}

static void backward(MEX_ARGS) {
  if (nrhs != 1) {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  plhs[0] = do_backward(prhs[0]);
}

static void update(MEX_ARGS) {
  if (solver_->net()) {
        //LOG(INFO) << "Begin update";
        solver_->ComputeUpdateValue();
        solver_->iter_++;
        //LOG(INFO) << "Compute updated values";
        net_->Update();
    //LOG(INFO) << "Network updated";    
  }
}

static void is_initialized(MEX_ARGS) {
  if (net_) {
    plhs[0] = mxCreateDoubleScalar(1);
  }else if (test_net_) {
    plhs[0] = mxCreateDoubleScalar(2);
  }else {
    plhs[0] = mxCreateDoubleScalar(0);
  }
}

static void get_device(MEX_ARGS) {
    if (nrhs != 0) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  Caffe::DeviceQuery();
}

static void set_iter(MEX_ARGS) {
    if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << "arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
    if (solver_) {
    solver_->iter_ = static_cast<int>(mxGetScalar(prhs[0]));
  }
}

static void read_mean(MEX_ARGS) {
    if (nrhs != 1) {
        mexErrMsgTxt("Usage: caffe('read_mean', 'path_to_binary_mean_file'");
        return;
    }
    const string& mean_file = mxArrayToString(prhs[0]);
    Blob<float> data_mean;
    LOG(INFO) << "Loading mean file from: " << mean_file;
    BlobProto blob_proto;
    bool result = ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);
    if (!result) {
        mexErrMsgTxt("Couldn't read the file");
        return;
    }
    data_mean.FromProto(blob_proto);
    mwSize dims[4] = {data_mean.width(), data_mean.height(),
                      data_mean.channels(), data_mean.num() };
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    caffe_copy(data_mean.count(), data_mean.cpu_data(), data_ptr);
    mexWarnMsgTxt("Remember that Caffe saves in [width, height, channels]"
                  " format and channels are also BGR!");
    plhs[0] = mx_blob;
}

/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "forward",            forward         },
  { "backward",           backward        },
  { "update",             update          },
  { "forward_test",       forward_test    },
  { "init",               init            },
  { "is_initialized",     is_initialized  },
  { "set_mode_cpu",       set_mode_cpu    },
  { "set_mode_gpu",       set_mode_gpu    },
  { "set_device",         set_device      },
  { "get_device",         get_device      },
  { "get_weights",        get_weights     },
  { "set_weights",        set_weights     },
  { "get_all_diff",       get_all_diff    },
  { "get_all_data",       get_all_data    },
  { "get_init_key",       get_init_key    },
  { "reset",              reset           },
  { "set_iter",           set_iter        },
  { "read_mean",          read_mean       },
  // The end.
  { "END",                NULL            },
};


/** -----------------------------------------------------------------
 ** matlab entry point: caffe(api_command, arg1, arg2, ...)
 **/
void mexFunction(MEX_ARGS) {
  mexLock();  // Avoid clearing the mex file.
  if (nrhs == 0) {
    mex_error("No API command given");
    return;
  }

  { // Handle input command
    char *cmd = mxArrayToString(prhs[0]);
    bool dispatched = false;
    // Dispatch to cmd handler
    for (int i = 0; handlers[i].func != NULL; i++) {
      if (handlers[i].cmd.compare(cmd) == 0) {
        handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
        dispatched = true;
        break;
      }
    }
    if (!dispatched) {
      ostringstream error_msg;
      error_msg << "Unknown command '" << cmd << "'";
      mex_error(error_msg.str());
    }
    mxFree(cmd);
  }
}
