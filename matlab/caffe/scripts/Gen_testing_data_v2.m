% v2 support inputs without image gradients
% use it when u use v2 for traning
function [batch, gt] = Gen_testing_data_v2( Solver, FILTER_TYPE, TEST_IMAGE)
img = imresize(im2double(TEST_IMAGE),[Solver.patchsize, Solver.patchsize]);
batch = zeros(Solver.patchsize, Solver.patchsize, 3, Solver.batchsize);
gt = batch;
switch FILTER_TYPE
    case 'L0'
        gt_img = L0Smoothing(img);
    case 'bilateral'
        for kk = 1:3
            gt_img(:,:,kk) = bilateralFilter(double(img(:,:,kk)),[],[],[],7,0.1,[],[]);
        end
    case 'RTV'
        gt_img = tsmooth(double(img),0.01,3);
    case 'WLS'
        for kk = 1:3
            gt_img(:,:,kk) = wlsFilter(double(img(:,:,kk)));
        end
    case 'WMF'
        gt_img = jointWMF(img,img,10,25.5,256,256,1,'exp');
    case 'shock'
        for kk = 1:3
            gt_img(:,:,kk) = shock(double(img(:,:,kk)));
        end
    otherwise
        error('Unknown method.')
end
batch(:,:,:,1) = img-0.5;
gt(:,:,:,1) = gt_img;
end
