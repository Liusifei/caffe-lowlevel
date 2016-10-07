function [dMy,dMx] = cal_diff(M)
dMy = zeros(size(M));dMx = zeros(size(M));
dMy(2:end,:,:,:) = diff(M,1,1);
dMx(:,2:end,:,:) = diff(M,1,2);