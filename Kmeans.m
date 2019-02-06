%%
% *Oskar Radermecker - CS413 Assignment : Problem 2 - Image Segmentation*

clear;
close all;

%im_name = input('Enter name of image :', 's');
lego = imread('lego-bricks-1.jpg'); %reads image

%% Noise

%nvar = [46.4 28.822 17.331];% Noise variability, respectively [SNR=1 SNR=10 SNR=20] 
%svar = var(double(lego(:)));% Signal variability

%A = size(lego); 
%noise =  normrnd(0, nvar(2), [A(1) A(2)]);
%lego = uint8(double(lego) + noise); %Add Noise

% %figure; imshow(lego); title('Initial Image. The image is being treated, please wait...'); %displays image
% %mse1 = immse(lego(:),legon(:)) %calculates the mean square value with legon = uint8(double(lego) + noise);
% %snr1 = 10*log(svar/mse1) %calculates the snr
% %lego = imgaussfilt(lego,2); %Gaussian filter --> worse restults

%% Initialisations

%clusters = input('Enter desired number of clusters :');
clusters = 5;

%% Algorithm

Total_number_objects = main_algo(lego, clusters);
h = msgbox(sprintf('Operation Completed. %d objects detected in total', Total_number_objects),'Success');

%% Functions 

function tot_obj = main_algo(img, clusters)
    
    clustered_image = cell(1,clusters);
    ord = cell(1,clusters);
    tot_obj = 0;

    space_labels = kmeans_clustering(img, clusters);

    for k = 1:clusters
        temp = img;
        temp(space_labels ~= k) = 0;
        clustered_image{k} = temp; %Creates clusters using the labels obtained by kmean algorithm
        ord{k} = uint8(zeros(1500, 3000,3)); pos = [70 70]; %Initialises new image for ordered objects and initial position
        [tobject, num_obj] = identify_objects(clustered_image{k}); %Identifies objects
        if num_obj < 40 %If too many objects, probably due to background cluster --> no real objects, only artifacts
            [~,new_order] = sort([tobject.Area]); %Finds new order, ordered by area
            object = tobject(new_order); %Orders objects
            tot_obj = tot_obj + num_obj; %Adds the number of objects to the total count
            figure(2*k-1); imshow(clustered_image{k}); hold on; %Displays cluster
            title(sprintf(['%d objects detected in cluster n°' num2str(k)],length(object)));
            for l = 1 : num_obj
                rectangle('Position', object(l).BoundingBox,'EdgeColor','g','LineWidth',1); %Prints Bounding Box
                angle = object(l).Orientation; MAL = object(l).MajorAxisLength;
                x_coord = object(l).Centroid(1) + [1 -1]*MAL*cosd(angle)/2;
                y_coord = object(l).Centroid(2) + [-1 1]*MAL*sind(angle)/2;
                line(x_coord, y_coord,'Color','g','LineWidth',1); %Prints Axis
                block = extract_block(clustered_image{k},object(l)); %Extracts the object
                [ord{k}, pos] = paste_in(ord{k},block,pos); %Pastes the object in the new image
            end
            figure(2*k); imshow(ord{k}); title(sprintf(['%d objects detected in cluster n°' num2str(k)],length(object)));
        end
    end
end

function [image, next_pos] = paste_in(image, block, upper_left_pos)
    D = size(block);
    image(upper_left_pos(1):upper_left_pos(1)+D(1)-1,upper_left_pos(2):upper_left_pos(2)+D(2)-1,:) = block;
    if upper_left_pos(2) < 2500
        next_pos = upper_left_pos + [0 D(2)+100];
    else
        next_pos = [upper_left_pos(1)+380 70];
    end
end

function block = extract_block(cluster, object) 
    imConvMask = object.ConvexImage; % Creates a filled mask for the object inside the region
    %imConvMask2 = object.FilledImage;
    imTemp2 = imcrop(cluster, [floor(object.BoundingBox(1)) floor(object.BoundingBox(2)) object.BoundingBox(3)-1 object.BoundingBox(4)]);%figure, imshow(imTemp2); 
    %imTemp = cluster(object.SubarrayIdx{:}); %crops the region from the cluster in b&w...
    s = size(imConvMask);
    block = zeros(s(1),s(2),3);
    for k=1:3
        block(:,:,k) = mat_mult(uint8(imConvMask),imTemp2(:,:,k));%clears the boundingbox of other blocks
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

    for k = 1:3
        temp1 = image(:,:,k);
        temp2 = imadjust(temp1); %increases contrast in temp3
        temp3 = imbinarize(temp2,graythresh(temp2));
        temp4 = imclose(temp3, strel('disk', 5));
        temp5 = imfill(logical(temp4), 'holes');
        finals(:,:,k) = bwareaopen(temp5,500); %removes too small components (have fewer than *argument* pixels)
    end
    finals = ( finals(:,:,1) + finals(:,:,2) + finals(:,:,3) ) > 0;
    objects = regionprops(bwconncomp(finals),'Area','Centroid','BoundingBox','ConvexImage','MajorAxisLength','Orientation');%,'SubarrayIdx','FilledImage');
    num_obj = length(objects);
end

function space_labels = kmeans_clustering(img, clusters) 

    lab_lego = applycform(img,makecform('srgb2lab')); %converting to L*a*b* color space (L = brightness, ab = colours) also : lab_lego = rgb2lab(lego);
    color_space = double(lab_lego(:,:,2:3));
    
    [rows, col, ~] = size(color_space);
    color_space = reshape(color_space,rows*col,2); %Reshapes h into a 2-columns vector (because 2 planes : a and b)
    
    idx = kmeans(color_space,clusters,'Replicates',3); %kmeans algorithm, repeated 3 times for better results
    
    plane_labels = reshape(idx,rows,col); %Reshapes the idx matrix (with the labelled, clustered pixels) in the dimension of original colour plane 
    space_labels = repmat(plane_labels,[1 1 3]); %replicates the label matrix for the three rgb colour planes (colour spaces) 
end

