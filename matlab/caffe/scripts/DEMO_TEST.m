% DEMO_TEST.m
addpath('../util');
addpath('../');
TEST_IMAGE = imread('org_0020.png');
model_path = '../models';
FILTER_TYPE = 'L0';
Solver = modelconfig(model_path, FILTER_TYPE, 'test');
out = Filter_Test(FILTER_TYPE, TEST_IMAGE, Solver);
% save out.mat out
imshow(out);title('Original/Proposed/GT');