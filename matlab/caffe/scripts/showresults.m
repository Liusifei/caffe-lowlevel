function [out,psnrs] = showresults(active, batch, gt)
o = double(batch) + 0.5;
g = double(gt);
r = double(active);
bz = size(o,4); psnrs = zeros(1,bz);
for m = 1:bz
psnrs(m) = psnr(r(:,:,:,m),g(:,:,:,m));
end
A = cat(2,o,r);
out = cat(2,A,g);
end