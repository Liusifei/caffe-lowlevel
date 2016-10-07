% Sifei Liu, 10/04/2016
% sliu32@ucmerced.edu
% Learn any type of image filters.
% FILTER_TYPE: supports the following methods:
%               'L0', 'shock', 'wls', 'WMF', 'RTV', 'RGF'
% TEST_IMAGE:  an rgb image.
% Solver:      solver configures and model parameters
% this version supports LRNN_v1.prototxt; in order to apply 
% LRNN_v2.prototxt u need to change "Gen_testing_data_v1" in line 26 to 
% "Gen_testing_data_v2"
function out = Filter_Test(FILTER_TYPE, TEST_IMAGE, Solver)
Solver = testdataconfig(Solver, TEST_IMAGE);
[batch, gt] = Gen_testing_data_v1( Solver, FILTER_TYPE, TEST_IMAGE );
batchc = {single(batch)};
fprintf('FP test image...\n');
active = caffe('forward_test',batchc, [Solver.patchsize,Solver.patchsize,5,Solver.batchsize]);
disp(length(active))
for c = 1:length(active)
    active_ = active{c};
    if size(active_,3) == 3
    [out,psnrs] = showresults(active_, batch(:,:,1:3,:), gt);
    end
end
fprintf('PSNR: %d\n',psnrs);
end