%
%   JointWMF - Joint-Histogram Weighted Median Filter 
%
%   O = jointWMF(I,F,r,sigma,nI,nF,iter,weightType) filter image "I" guided
%   by feature map "F". The result value of each pixel is the weighted median
%   of its neigbouring pixels in a local window with radius "r". The weight
%   is defined as the affinity between the corresponding pixels on image "F".
%   The affinity is also controlled by a value "sigma". If "weightType"
%   equals 'exp', the weight of pixel p and q is exp(-|F(p)-F(q)|^2/(2*sigma^2)).
%   
%   Paras: 
%   @I         : Input image, UINT8 or DOUBLE image, any # of channels
%   @F         : Feature image, UINT8 or DOUBLE image, 1 or 3 channels
%   @r         : window radius. 
%                Range [0,inf), 10 by default.
%   @sigma     : sigma controls the weight between two pixels. When
%                "weightType" equals 'exp', sigma is the standard deviation
%                of the Gaussian kernel. 
%                Range (0, inf), 25.5 by defalut.  
%   @nI        : Number of possible input values (per channel). For UINT8 input,
%                "nI"=256. For DOUBLE input, if the input values have more than 
%                "nI" possibilities, they will be adaptively quantized into "nI" values. 
%                Range (0, inf), 256 by defalut.
%   @nF        : Number of possible feature values. For single channel
%                UINT8 input, "nF"=256. In other cases, if the input values
%                have more than "nF" possibilities, they will be adaptively
%                quantized into "nF" values.
%                Range (0, inf), 256 by defalut.
%   @iter      : # of iterations. If iter > 1, the filtering process will
%                be iteratively repeated without changing the feature map "F".  
%                Range [0,inf), 1 by defalut.   
%   @weightType: The weight form that defines the pixel affinity of pixel "p" and "q".
%                 'exp': exp(-|F(p)-F(q)|^2/(2*sigma^2)) [default]
%                 'iv1': (|F(p)-F(q)|+sigma)^-1 
%                 'iv2': (|F(p)-F(q)|^2+sigma^2)^-1
%                 'cos': dot(F(p),F(q))/(|F(p)|*|F(q)|)
%                 'jac': (min(r1,r2)+min(g1,g2)+min(b1,b2))/(max(r1,r2)+max(g1,g2)+max(b1,b2))
%                        where F(p)=(r1,g1,b1) and F(q)=(r2,g2,b2)
%                 'off': 1
%   @mask      : Mask image, UINT8 or DOUBLE type, 1 channel. This argument is to
%                disable the influence of some pixels, which is usable for
%                handling occlusion in optical flow estimation. The pixel
%                with mask 0 will not be counted in joint-histogram.
%
%   Example
%   ==========
%   I = im2double(imread('image.png'));
%   O = jointWMF(I,I);
%   figure, imshow(O);
%
%
%   Note
%   ==========
%   To achieve higher accuracy of our approximation: 
%       For floating point input, you may increase nI, e.g. 1024, 2048. 
%       For three channel feature, you may increase nF, e.g. 512, 1024.
%
%   ==========
%   The Code is created based on the method described in the following paper:
%   [1] "100+ Times Faster Weighted Median Filter", Qi Zhang, Li Xu, Jiaya Jia, IEEE Conference on 
%		Computer Vision and Pattern Recognition (CVPR), 2014
%
%   The code and the algorithm are for non-comercial use only.
%
%   Due to the adaption for supporting mask and different types of input, this code is
%   slightly slower than the one claimed in the original paper. Please use
%   our executable on our website for performance comparison.
%  
%   Author: Qi Zhang (zhangqi@cse.cuhk.edu.hk)
%   Date  : 09/21/2014
%   Version : 1.1 
%   Copyright 2014, The Chinese University of Hong Kong.
% 

function O = jointWMF(I,F,r,sigma,nI,nF,iter,weightType,mask)

%% validation for I %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% handle empty I
assert(exist('I','var')==1,'Error: I can not be empty.');

% ensure I belongs to double or uint8
if (~strcmp(class(I),'double'))&&(~strcmp(class(I),'uint8'))
   I = double(I); 
end


%% validation for F %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% handle empty F
if ~exist('F','var')
    F = I;
end

% ensure # of channels
assert(size(F,3)==1 || size(F,3)==3,'Error: F should have 1 or 3 channels.');

% ensure F belongs to uint8
if ~strcmp(class(F),'uint8')
    
    % for other types, do normalization and quantization
    F = double(F);
    maxv = max(max(max(F)));
    minv = min(min(min(F)));
    range = maxv - minv;
    
    F = F-minv;
    if range ~= 0
        F = F.*(255.0/range);
        if exist('sigma','var')
            sigma = sigma*(255.0/range);
        end
    end
    
    F = uint8(F); 
end

%% validation for r %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% handle empty r
if ~exist('r','var')
    r = 10;
end

%% validation for sigma %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% handle empty sigma
if ~exist('sigma','var')
    sigma = 25.5;
end

%% validation for nI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% handle empty nI
if ~exist('nI','var')
    nI = 256;
end

%% validation for nF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% handle empty nF
if ~exist('nF','var')
    nF = 256;
end

%% validation for iter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% handle empty iter
if ~exist('iter','var')
    iter = 1;
end

%% validation for weightType %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% handle empty weightType
if ~exist('weightType','var')
    weightType = 'exp';
end

%% validation for mask %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% handle empty mask
if ~exist('mask','var')
    mask = uint8(ones(size(I,1),size(I,2)));
end

% mask must be uint8 1-channel
mask = uint8(mask(:,:,1));

O = mexJointWMF(I,F,r,sigma,nI,nF,iter,weightType,mask);

if strcmp(class(I),'uint8')
    O = uint8(O);
end


end