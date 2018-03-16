clear;
rgb_image=imread('0.jpg');  
gray_image=rgb2gray(rgb_image);  
figure(1); 
imshow(gray_image);

figure(2); 
[m,n] = size(gray_image); 
dst_bw = zeros(m,n);
n_split = 10;
num=1;
for i = 1:n_split
    for j = 1:n_split
        m_start=1+(i-1)*fix(m/n_split);
        m_end=i*fix(m/n_split);
        n_start=1+(j-1)*fix(n/n_split);
        n_end=j*fix(n/n_split);
        split_gray_image=gray_image(m_start:m_end,n_start:n_end,:); %将每块读入矩阵
        level = graythresh(split_gray_image);  
        bw=im2bw(split_gray_image,level); 
        dst_bw(m_start:m_end,n_start:n_end,:)=bw;
        num=num+1;
    end
end 
imshow(dst_bw);