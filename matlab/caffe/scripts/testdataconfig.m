% parameters should be consistent with prototxt
% patchsize: width and height in caffe
% batchsize: num in caffe, can be changed accordingly
% supporting JPG and PNG only
function Solver = testdataconfig( Solver, img )
% Solver.height = size(img, 1);
% Solver.width = size(img, 2);
Solver.patchsize = 256;
Solver.batchsize = 1;
end