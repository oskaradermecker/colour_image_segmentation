%%
% *Oskar Radermecker - CS413 Assignment : Problem 2 - Image Segmentation*

clear;
close all; 

lego = imread('lego-bricks-1.JPG'); %reads image

[r, g, ~] = size(lego);
Final_image = zeros(r,g,3);

for k = 1:3
    Final_image(:,:,k) = sob_edge_det(lego(:,:,k));
end
Final_image=(Final_image(:,:,1)+Final_image(:,:,2)+Final_image(:,:,3));
final = zwz(lego, Final_image, r, g); 
figure, imshow(final);

%% Functions

function fin_temp = zwz(orig_img,labels, r, g)
fin_temp = orig_img;
for i=1:r
    for j=1:g
        if labels(i,j) == 0
            fin_temp(i,j,1) = 0;
            fin_temp(i,j,2) = 0;
            fin_temp(i,j,3) = 0;
        end
    end
end
end

function final = sob_edge_det(in_image)

    se90 = strel('line', 3, 90);
    se0 = strel('line', 3, 0);
    seD = strel('diamond',1);
    
    [~, threshold] = edge(in_image,'sobel');
    temp1 = edge(in_image,'sobel', threshold * 0.5);
    temp2 = imdilate(temp1, [se90 se0]);
    temp3 = imclose(temp2,se0);
    temp4 = imfill(temp3, 'holes');
    temp5 = imclearborder(temp4, 4);
    temp6 = activecontour(temp5, bwconvhull(temp5, 'Union'), 500, 'edge');
    
    
    final = imerode(temp6,seD);
    final = imerode(final,seD);
    final = imerode(final,seD);
    figure, imshow(final), title('segmented image');
end