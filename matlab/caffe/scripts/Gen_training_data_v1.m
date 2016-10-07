% v1 support inputs with image gradients (x and y)
function [batch, gt] = Gen_training_data_v1( Solver,  FILTER_TYPE)
batch = single(zeros(Solver.patchsize,Solver.patchsize,5,Solver.batchsize));
gt = single(zeros(Solver.patchsize,Solver.patchsize,3,Solver.batchsize));
rng('shuffle');
idpool = randperm(Solver.train_num);
count = 1;
while count <= Solver.batchsize
    idx = idpool(count);
    img = im2single(imread(fullfile(Solver.train_folder,Solver.trainlst{idx})));
    [r,c,cn] = size(img);
    if cn == 1
        img = repmat(img, [1 1 3]);
    end
    if min(r,c) > 1000
        img = imresize(img,1000/min(r,c));
        [r,c,~] = size(img);
    end
    rate =  (rand - 0.5)/5;
    shift_x = floor(max(r,c) * rate);
    rate =  (rand - 0.5)/5;
    shift_y = floor(max(r,c) * rate);
    scale_x = 1+(rand-0.5)/5;
    scale_y =  scale_x;
    angle = (rand - 0.5)*(30/180)*pi;
    A = [scale_x * cos(angle), scale_y * sin(angle), shift_x;...
        -scale_x * sin(angle), scale_y * cos(angle), shift_y]';
    T = maketform('affine', A);
    simg = single(imtransform(img, T, 'XYScale',1));
    [r,c,~] = size(simg);
    grad = 0; ite = 0;
    while grad < 1.5e3 && ite < 10
        ite = ite + 1;
        if min(r,c) < Solver.patchsize
            img = imresize(simg,(Solver.patchsize/min(r,c)+0.01));
            roi = img(1:1+Solver.patchsize-1,1:1+Solver.patchsize-1,:);
        else
            margin_x = c - Solver.patchsize;
            margin_y = r - Solver.patchsize;
            rand_x = ceil(rand * margin_x); rand_y = ceil(rand * margin_y);
            roi = simg(max(1,rand_y):max(1,rand_y)+Solver.patchsize-1, max(1,rand_x):max(1,rand_x)+Solver.patchsize-1,:);
        end
        [gx,gy] = gradient(roi); grad = unique(sum(sum(sum(abs(gx)+abs(gy)))));
    end
    switch FILTER_TYPE
        case 'L0'
            gt_img = L0Smoothing(roi);
        case 'bilateral'
            for kk = 1:3
                gt_img(:,:,kk) = bilateralFilter(double(roi(:,:,kk)),[],[],[],7,0.1,[],[]);
            end
        case 'RTV'
            gt_img = tsmooth(double(roi),0.01,3);
        case 'WLS'
            for kk = 1:3
                gt_img(:,:,kk) = wlsFilter(double(roi(:,:,kk)));
            end
        case 'WMF'
            gt_img = jointWMF(roi,roi,10,25.5,256,256,1,'exp');
        case 'shock'
            for kk = 1:3
                gt_img(:,:,kk) = shock(double(roi(:,:,kk)));
            end
        otherwise
            error('Unknown method.')
    end
    gray_roi = rgb2gray(roi);
    [dx,dy] = gradient(gray_roi);
    batch(:,:,1:3,count) = roi-0.5;
    batch(:,:,4,count) = dx;
    batch(:,:,5,count) = dy;
    gt(:,:,:,count) = gt_img;
    count = count + 1;
end
end