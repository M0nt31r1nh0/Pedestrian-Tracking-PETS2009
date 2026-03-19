clear all; close all; clc;

gt = readmatrix('PETS-S2L1\\gt\\gt.txt');  %read the ground truth

numFramesBG = 794;  %better results with 794, but faster with lower number of frames to do the median

for i = 1: numFramesBG
    img = imread(sprintf('Crowd_PETS\\S2\\L1\\Time_12-34\\View_001\\frame_%.4d.jpg', i));
    
    if i == 1
        [h, w, c] = size(img);
        stack = zeros(h, w, c, numFramesBG, 'double');
    end
    
    stack(:, : ,: , i) = double(img);
end

imgbk = uint8(median(stack, 4));    %estimation of the background

thr = 40;
minArea = 100;
baseValue = 794;

maxDist = 50;
maxTrail = 20;
trajectories = {}; 
trajCount = 0;

se = strel('disk', 2);


figure; hold on
for i = 0: baseValue
    frameNumber = i + 1;
    imgfr = imread(sprintf('Crowd_PETS\\S2\\L1\\Time_12-34\\View_001\\frame_%.4d.jpg', frameNumber));

    imgdif = (abs(double(imgbk(:, :, 1)) - double(imgfr(:, :, 1))) > thr) | ...
             (abs(double(imgbk(:, :, 2)) - double(imgfr(:, :, 2))) > thr) | ...
             (abs(double(imgbk(:, :, 3)) - double(imgfr(:, :, 3))) > thr);


    bw1 = imgdif;
    bw2 = imclose(bw1, se);
    bw3 = imerode(bw2, se);

    bwR = bw3;

    [lb, num] = bwlabel(bwR);
    regionProps = regionprops(lb, 'Area', 'BoundingBox');

    inds = find([regionProps.Area] > minArea);

    imshow(imgfr);
    hold on;

    gt_frame = gt(gt(:,1) == frameNumber, :);

    for k = 1:size(gt_frame,1)
        x = gt_frame(k,3);
        y = gt_frame(k,4);
        w = gt_frame(k,5);
        h = gt_frame(k,6);
        conf = gt_frame(k,7);

        if conf > 0
            rectangle('Position', [x y w h], ...
                      'EdgeColor', 'g', ...
                      'LineWidth', 2);
        end
    end

    currCentroids = [];

    for j = inds
        bbox = regionProps(j).BoundingBox;

        rectangle('Position', bbox, ...
                  'EdgeColor', 'y', ...
                  'LineWidth', 2);

        cx = bbox(1) + bbox(3)/2;
        cy = bbox(2) + bbox(4)/2;

        currCentroids = [currCentroids; cx cy];

        text(cx, cy, sprintf('%d', j), ...
        'Color', 'y', ...
        'FontSize', 12, ...
        'FontWeight', 'bold');
    end

    if isempty(trajectories)
        for k = 1:size(currCentroids,1)
            trajCount = trajCount + 1;
            trajectories{trajCount} = currCentroids(k,:);
        end
    else
        assigned = zeros(size(currCentroids,1),1);
    
        for t = 1:length(trajectories)
            lastPos = trajectories{t}(end,:);
    
            if isempty(currCentroids)
                continue;
            end
    
            dists = sqrt(sum((currCentroids - lastPos).^2, 2));
    
            [minDist, idxMin] = min(dists);
    
            if minDist < maxDist && ~assigned(idxMin)
                trajectories{t} = [trajectories{t}; currCentroids(idxMin,:)];
                assigned(idxMin) = 1;
            end
        end
    
        for k = 1:size(currCentroids,1)
            if ~assigned(k)
                trajCount = trajCount + 1;
                trajectories{trajCount} = currCentroids(k,:);
            end
        end
    end

    for t = 1:length(trajectories)
        traj = trajectories{t};

        if size(traj,1) > 1
            startIdx = max(1, size(traj,1) - maxTrail);
            plot(traj(startIdx:end,1), traj(startIdx:end,2), ...
                 'r-', 'LineWidth', 2);
        end
    end

    hold off;
    drawnow;
end