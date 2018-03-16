clear;
rgb = imread('10.jpg');
figure(1); 
imshow(rgb);

lab = rgb2lab(rgb);

figure(2); 
subplot(3,1,1);
imshow(lab(:,:,1),[0 100]);

subplot(3,1,2);
imshow(lab(:,:,2),[-120 120]);

subplot(3,1,3);
imshow(lab(:,:,3),[-120 120]);