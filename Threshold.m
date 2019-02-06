%%
% *Oskar Radermecker - CS413 Assignment : Problem 2 - Image Segmentation*

clear; 
close all;

rgb = imread('lego-bricks-1.jpg'); %read image

%% Noise

%nvar = [46.4 28.822 17.331];% Noise variability, respectively [SNR=1 SNR=10 SNR=20] 
%svar = var(double(rgb(:)));% Signal variability

%A = size(rgb); 
%noise =  normrnd(0, nvar(2), [A(1) A(2)]);
%rgb = uint8(double(rgb) + noise); %Add Noise

% %figure; imshow(rgb); title('Initial Image. The image is being treated, please wait...'); %displays image
% %mse1 = immse(rgb(:),rgbn(:)) %calculates the mean square value
% %snr1 = 10*log(svar/mse1) %calculates the snr
% %rgb = imgaussfilt(rgb,2); %Gaussian filter --> worse restults

%% Initialisations

%Colour Properties (thresholds for each colour plane)

%For lego-bricks
col_prop = struct('name',{'Red','Yellow','Green','Cyan','Dark Blue','Black'},'hueLow',...
    {0.035,0.095,0.17,0.39,0.54,0.01},'hueHigh',{0.9,0.13,0.31,0.5,1,0.13},'saturationLow',...
    {0.6,0.87,0.36,0.2,0,0.45},'saturationHigh',{1,1,1,0.5,1,1},'valueLow',...
    {0.42,0.4,0,0.1,0,0},'valueHigh',{1,1,0.45,0.7,1,0.35},'redornot',{1,0,0,0,0,0});

%For M&Ms
% col_prop = struct('name',{'Red','Yellow','Green','Blue','Brown','Orange'},'hueLow',...
%     {0.05,0.128,0.28,0.563,0.754,0.05},'hueHigh',{1,0.183,0.401,0.657,0.225,0.126},'saturationLow',...
%     {0.4,0.473,0,0.253,0,0.55},'saturationHigh',{0.7,1,1,1,0.341,1},'valueLow',...
%     {0,0,0,0,0,0.207},'valueHigh',{1,1,1,1,0.17,1},'redornot',{1,0,0,0,1,0});

%% Algorithm

Total_number_objects = main_algo(col_prop, rgb);
h = msgbox(sprintf('Operation Completed. %d objects detected in total', Total_number_objects),'Success');

%% Functions 

function tot_obj = main_algo(col_prop, image)
    
    size_col_prop = size(col_prop);
    iso_colour = cell(1,size_col_prop(2)); ord = cell(1,size_col_prop(2)); 
    tot_obj = 0;
    
    for k=1:size_col_prop(2)
        iso_colour{k} = threshold(image,col_prop(k).hueLow,col_prop(k).hueHigh,col_prop(k).saturationLow,...
            col_prop(k).saturationHigh,col_prop(k).valueLow,col_prop(k).valueHigh,col_prop(k).redornot); %Threshold Image
        ord{k} = uint8(zeros(1500, 3000,3)); pos = [70 70]; %Initialises new image for ordered objects and initial position
        [tobject, num_obj] = identify_objects(iso_colour{k}); %Identifies objects
        [~,new_order] = sort([tobject.Area]); %Finds new order, ordered by area
        object = tobject(new_order); %Orders objects
        tot_obj = tot_obj + num_obj; %Adds the number of objects to the total count
        figure(2*k-1); imshow(iso_colour{k}); hold on; %Displays cluster
        title(sprintf('%d objects detected in %s colour cluster',length(object),col_prop(k).name));
        for l = 1 : num_obj
            rectangle('Position', object(l).BoundingBox,'EdgeColor','g','LineWidth',1); %Prints Bounding Box
            angle = object(l).Orientation; MAL = object(l).MajorAxisLength; 
            x_coord = object(l).Centroid(1) + [1 -1]*MAL*cosd(angle)/2;
            y_coord = object(l).Centroid(2) + [-1 1]*MAL*sind(angle)/2;
            line(x_coord, y_coord,'Color','g','LineWidth',1); %Prints Axis
            block = extract_block(iso_colour{k},object(l)); %Extracts the object
            [ord{k}, pos] = paste_in(ord{k},block,pos); %Pastes the object in the new image
        end
        figure(2*k); imshow(ord{k}); title(sprintf('%d objects detected in %s colour cluster',length(object),col_prop(k).name));
    end
end

function [image, next_pos] = paste_in(image, block, upper_left_pos)
    D = size(block);
    image(upper_left_pos(1):upper_left_pos(1)+D(1)-1,upper_left_pos(2):upper_left_pos(2)+D(2)-1,:) = block;
    if upper_left_pos(2) < 2500 %Maximum position after which it is necessary to change line
        next_pos = upper_left_pos + [0 D(2)+100];
    else
        next_pos = [upper_left_pos(1)+380 70];
    end
end

function block = extract_block(cluster, object)
    imConvMask = object.ConvexImage; % Creates a filled mask for the object inside the region
    imTemp2 = imcrop(cluster, [floor(object.BoundingBox(1)) floor(object.BoundingBox(2)) object.BoundingBox(3)-1 object.BoundingBox(4)]); %Crops the boundingbox with the object from the cluster 
    s = size(imConvMask);
    block = zeros(s(1),s(2),3);
    for k=1:3
        block(:,:,k) = mat_mult(uint8(imConvMask),imTemp2(:,:,k));%clears the boundingbox from other blocks
    end  
    block = imrotate(block,-90-object.Orientation); %rotates the image
end

function mat = mat_mult(mat1, mat2) 
    s1 = size(mat1); s2 = size(mat2);
    mat = mat1;
    minl = min(s1(1),s2(1)); minc = min(s1(2),s2(2));
    for r = 1:minl
       for c = 1:minc
           mat(r,c) = mat1(r,c) .* mat2(r,c);
       end
    end
end

function [objects, num_obj] = identify_objects(image)
    [r,g,~] = size(image);
    finals = zeros(r,g,3);

    for k = 1:3 %Work on the three planes
        temp1 = image(:,:,k);
        temp2 = imadjust(temp1); %increases contrast in temp3
        temp3 = imbinarize(temp2,graythresh(temp2)); %binarises the image using Otsu's method to identify the threshold
        temp4 = imclose(temp3, strel('disk', 5)); %fills gaps and smooths edges
        temp5 = imfill(logical(temp4), 'holes'); %fills the remaining gapses
        finals(:,:,k) = bwareaopen(temp5,500); %removes components which have fewer than 500 pixels
    end
    finals = ( finals(:,:,1) + finals(:,:,2) + finals(:,:,3) ) > 0;
    objects = regionprops(bwconncomp(finals),'Area','Centroid','BoundingBox','ConvexImage','MajorAxisLength','Orientation');
    num_obj = length(objects);
end

function [thresholded_image] = threshold(rgb_image,lhue,hhue,lsaturation,hsaturation,lvalue,hvalue,redornot)
    hsv = rgb2hsv(rgb_image); %image transformed to HSV Space
    thresholded_image = rgb_image;
    
    if redornot
        hue_thresh = (hsv(:,:,1) >= hhue) | (hsv(:,:,1) <= lhue);
    else
        hue_thresh = (hsv(:,:,1) <= hhue) & (hsv(:,:,1) >= lhue);
    end
    saturation_thresh = (hsv(:,:,2) <= hsaturation) & (hsv(:,:,2) >= lsaturation);
    value_thresh = (hsv(:,:,3) <= hvalue) & (hsv(:,:,3) >= lvalue);
    
    total_thresh = hue_thresh & saturation_thresh & value_thresh;
    total_thresh = uint8(bwareaopen(total_thresh, 100));
    total_thresh = imclose(total_thresh, strel('disk', 5));
    total_thresh = imfill(logical(total_thresh), 'holes');

    thresholded_image(repmat(~total_thresh,[1 1 3])) = 0;
end