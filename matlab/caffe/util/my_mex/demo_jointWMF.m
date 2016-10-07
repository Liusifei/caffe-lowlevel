
close all;

I = im2double(imread('imgs/image1.png'));

tic;
res = jointWMF(I,I,10,25.5,256,256,1,'exp');
toc;

imwrite(res,'result.png');
%figure, imshow(res);
