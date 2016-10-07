function [delta, loss] = L2Loss_hardsample(active, gt, mode, sparse)
if ~exist('sparse','var')
   sparse = 0;
end
[r,c,cha,bz] = size(active);
if size(gt,1)~= r
    gt = imresize(gt,[r,c]);
end
loss = zeros(2,1);
dt = active - gt;
if sparse
   alpha = 0.1;
   dg = zeros(r,c,cha,bz);
   for kk = 1:cha
    [dy,dx] = cal_diff(active(:,:,kk,:));
    [dloss_,dg_] = GradSparse(dy,dx);
    if kk == 1
      dloss = dloss_;
    else
      dloss = dloss + dloss_;
    end
    dg(:,:,kk,:) = dg_;
   end
   loss(1) = 0.5 * sum(dt(:).^2)/bz;
   loss(2) = sum(dloss(:))/bz;
else
   loss(1) = 0.5 * sum(dt(:).^2)/bz;
end
if strcmp(mode, 'train')
    msk1 = hardsample(dt, [1:cha], [r,c,cha,bz], 0.4);
    msk2 = uni_balance([r,c,cha,bz], 0.1);
    msk = max(cat(5,msk1,msk2),[],5);
    if sparse
       delta = single((msk.*dt + alpha * dg)/bz);
    else
       delta = single((msk.*dt)/bz);
    end
else
    delta = 0;
end
end

function mask = uni_balance(dsize, rate)
mask = zeros(dsize);
for m = 1:dsize(4)
    msk = rand(dsize(1),dsize(2)) < rate;
    mask(:,:,:,m) = repmat(msk,[1,1,dsize(3)]);
end
end

function mask = hardsample(delta, channel, dsize, rate)
mask = zeros(dsize);
num = round(rate * (dsize(1)*dsize(2)));
for m = 1:dsize(4)
    msk = zeros(dsize(1),dsize(2));
    det = sum(abs(delta(:,:,:,m)),3);
    [Y,I] = sort(det(:), 1, 'descend');
    msk(I(1:num)) = 1;
    mask(:,:,channel,m) = repmat(msk,[1,1,length(channel)]);
end
end

function [loss,delta] = GradSparse(dy, dx)
[r,c,~,bz] = size(dy);
delta = zeros(r,c,1,bz);
for m = 1:bz
    dy_ = dy(:,:,1,m); dx_ = dx(:,:,1,m); 
    loss(m) = sum(sqrt(dx_(:).^2+1e-5))+sum(sqrt(dy_(:).^2+1e-5));
    ddy = zeros(size(dy_));
    ddy(2:end,:) =  diff(dy_,1,1);
    ddx = zeros(size(dx_));
    ddx(:,2:end) =  diff(dx_,1,2);
    %delta(:,:,1,m) = ddy./sqrt(dy_.^2+1e-5) + ...
     %                ddx./sqrt(dx_.^2+1e-5);
    delta(:,:,1,m) = ddy/sum(abs(dy_(:))) + ddx/sum(abs(dx_(:)));
end
end
