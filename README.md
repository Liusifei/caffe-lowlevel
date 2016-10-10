
## ECCV-Learning Recursive Filter for Low-Level Vision via a Hybrid Network

Released in 10/10/2016. The codes are based on [caffe](https://github.com/BVLC/caffe).

## Description

This is a demo implementation for ECCV16 paper: Learning Recursive Filter for Low-Level Vision via a Hybrid Network. We provide both training and testing demo, as well as the pretrained models for various image filters, including: L0, bilateral, related total variation filter (RTV), weighted leaset square (WLS), weighted median filter (WMF) and shock filter. Other filters can be also created through the same architecture by users.
We will introduce the layers for linear recurrent network and the main script in the following sections.

## Layers
The four-directional linear recurrent networks are done by a newly implemented layer “class GateRecurrentLayer”, with the type of “GateRecurrent". The users can either clone the whole package, or added the relative declaration in hpp, the “gaterecurrent_layer.cpp/cu”, as well as the relative id and message from “caffe.proto” into their own caffe package.
The basic configuration in prototxt is:
```
layer {
  name:"rnn1"
  type:"GateRecurrent"
  bottom:"data"
  bottom:"gate"
  top:"rnn1"
  param{
     lr_mult: 0.1
     decay_mult: 1
   }
   param {
     lr_mult: 0.01
     decay_mult: 0
   }
  gaterecurrent_param {
     horizontal: true
     reverse: false
     active: LINEAR
     num_output: 16
     use_bias: false
     use_wx: false
     use_wh: false
     restrict_g: 1.0
     restrict_w: 0.5
     weight_filler {
       type: "xavier"
     }
     bias_filler {
       type: "constant"
       value: 0
     }
   }
}
```
In the original implementations, the convolutional weight matrix and the bias, which are similar to the vanilla RNN are also implemented. Since they are not used in the paper, the parameters of “use_bias”, “use_wx”, “use_wh”, “restrict_w” are not used. Similarly, the fields of “param”, “weight_filler”, “bias_filler” can be left as default since not been used either. The “restrict_g” should set to 1 to make the system stable (there is a TANH for the topmost layer of gate, which also aims to keep the system stable), and the active should set to “LINEAR” in this work. 

This layer can also be used as spatial vanilla RNN using different settings, however, it is beyond the scope of this work.
The parallel and cascaded connections are illustrated as follows, more examples can be found in matlab/caffe/models.

### Parallel connection
```
layer { name: “rnn1” type: “GateRecurrent” bottom: “data” bottom: “gate1” top: “rnn1”…}
layer { name: “rnn2” type: “GateRecurrent” bottom: “rnn1” bottom: “gate2” top: “rnn2”…}
```
### Cascade connection
```
layer { name: “rnn1” type: “GateRecurrent” bottom: “data” bottom: “gate1” top: “rnn1”…}
layer { name: “rnn2” type: “GateRecurrent” bottom: “data” bottom: “gate2” top: “rnn2”…}
…
layer {
  name: "EltwiseSum1"
  type: "Eltwise"
  bottom: "rnn1"
  bottom: "rnn2"
  top: "rnn12"
  eltwise_param {
      operation: SUM
  }
}
```

## Scripts and pre-trained models
All scripts are developed using matlab scripts, and can be found in matlab/caffe/scripts. Users can use DEMO_TRAIN to learn their own models. For training other filters that are not provided, please put the existing filter functions in matlab/caffe/utils.
To use the pre-trained models, users should first download them using get_model.sh in matlab/caffe/models, and then use DEMO_TEST to generate the results of any images. Note that the shock filter has different architectures with the others, using LRNN_v2.prototxt, so one should change the LRNN_solver.prototxt first in training/applying the pre-trained model. 
An additional net LRNN_v3.prototxt is provided, which can replace the v1 and achieve the similar performance with even more efficient structure. Note that we do not provide the pre-trained model for this version, however, users can obtain the model simply by training this model using DEMO_TRAIN, without adjusting any solver parameter.
User should put their own training set path, which can be any kind of images, in DEMO_TRAIN. Usually 10 thousands random images are far enough to train a good model.

## Citations
Please cite this paper in your publications if it helps your research:
```
@inproceedings{liu2016learning,
  title={Learning Recursive Filters for Low-Level Vision via a Hybrid Neural Network},
  author={Sifei Liu and Jinshan Pan and Ming-Hsuan Yang},
  booktitle={European Conference on Computer Vision},
  year={2016},
}
```
