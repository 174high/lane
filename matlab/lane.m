clear;
rgb_image=imread('timg.jpg');  
gray_image=rgb2gray(rgb_image);  
figure(1); 
imshow(gray_image);

level = graythresh(gray_image);  
bw=im2bw(gray_image,level); 
figure(2); 
imshow(bw);