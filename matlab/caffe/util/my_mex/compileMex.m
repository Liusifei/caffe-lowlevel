
%mex -setup

include = '-I./include';
libPath = './lib';
lib1 = fullfile(libPath,'libopencv_core.so');

mex('mexJointWMF.cpp',include,lib1);
