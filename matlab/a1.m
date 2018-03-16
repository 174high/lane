clear all;  
close all ;  
I=imread('1.jpg');  
G=rgb2gray(I); 
figure();  

subplot(2,2,1);  
imshow(G);  

subplot(2,2,2);  
imhist(G);  

subplot(2,2,3);  
imhist(G);  
[h,x]=imhist(G);  
h=smooth(h,7);  
plot(x,h)  
%求出阈值T  
df1=diff(h);%一阶差分  
df2=diff(df1);%二阶差分  
[m,n]=size(df2);  
T=0;  
for i=1:m  
if(abs(df1(i+1))<=0.15 && df2(i)>0)  
    T=x(i+2)%确定阈值  
    break;  
end  
end  
G=im2bw(G,T/255);%转为二值图像  
subplot(2,2,4);  
imshow(G);  