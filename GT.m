clear all; close all; 

disp('Starting background extraction...')

gt = readmatrix('PETS-S2L1\gt\gt.txt');
numFramesBG = 795;
numFramesProcess = 795;

bgIdx = round(linspace(0, numFramesBG-1, 40));
bgIdx = unique(bgIdx);

for n = 1:length(bgIdx)
    frameIdx = bgIdx(n);
    img = imread(sprintf('Crowd_PETS\\S2\\L1\\Time_12-34\\View_001\\frame_%.4d.jpg', frameIdx));

    if n == 1
        [h, w, c] = size(img);
        stack = zeros(h, w, c, length(bgIdx), 'double');
    end

    stack(:,:,:,n) = double(img);
end

imgbk = uint8(median(double(stack), 4));

disp('Completed background extraction.')

thr      = 40;
minArea  = 300;


se = strel('disk', 2);

maxDist    = 50;
trackCount = 0;
maxTrail   = 20;
maxMissed  = 10;   

tracks = struct( ...
    'id',           {}, ... % Track ID
    'centroid',     {}, ... % Current position
    'trajectory',   {}, ... % History of positions for drawing trails
    'hist',         {}, ... % Appearance histogram for re-identification
    'missedFrames', {}, ... % Counter for consecutive missed detections
    'activeFrames', {}, ... % Counter for consecutive active frames (for confidence estimation)
    'velocity',     {}, ... % Estimated velocity
    'bbox',         {}, ... % Latest bounding box for visualization and output
    'isActive',     {}, ...  % Flag to indicate if track is currently active (matched in current frame or recently)
    'confidence',   {}  ... % Confidence score for the track 
);

% Clear old results file before writing new results
if exist('results.txt', 'file')
    delete('results.txt');
end

% Open a file to save detection/tracking results
outFile = fopen('results.txt', 'w');
if outFile == -1
    error('Cannot open output file results.txt for writing.');
end

disp('Starting algorithm...')


figure;

for frameNumber = 1:numFramesProcess

    imgfr = imread(sprintf('Crowd_PETS\\S2\\L1\\Time_12-34\\View_001\\frame_%.4d.jpg', frameNumber-1));
    imgdif = (abs(double(imgbk(:,:,1)) - double(imgfr(:,:,1))) > thr) | ...
             (abs(double(imgbk(:,:,2)) - double(imgfr(:,:,2))) > thr) | ...
             (abs(double(imgbk(:,:,3)) - double(imgfr(:,:,3))) > thr);

    
    bw1 = imgdif;
    bw2 = imclose(bw1, se);
    bw3 = imerode(bw2, se);
    bwR = bw3;

    [lb, num] = bwlabel(bwR);
    stats = regionprops(lb, 'Area', 'BoundingBox', 'Image');
    inds = find([stats.Area] > minArea);

    
    currentCentroids = [];
    currentHists     = {};
    currentBBoxes    = {};

    for idx = 1:length(inds)
        j    = inds(idx);
        bbox = stats(j).BoundingBox;


        cx = bbox(1) + bbox(3)/2;
        cy = bbox(2) + bbox(4)/2;

        blobMask = stats(j).Image;
        hist_val = computeHistogram(imgfr, bbox, blobMask);

        currentCentroids          = [currentCentroids; cx cy];
        currentHists{end+1}       = hist_val;
        currentBBoxes{end+1}      = bbox;
    end

    assignedDets   = zeros(size(currentCentroids,1), 1);
    assignedTracks = zeros(length(tracks), 1);

    %try to asign an existing track to each detection based on predicted position and histogram similarity
    for t = 1:length(tracks)
        if ~tracks(t).isActive
            continue; % Skip inactive tracks
        end
        lastHist = tracks(t).hist;

        bestCost = 0.3;
        bestIdx   = -1;

        
        for k = 1:size(currentCentroids,1)
            if assignedDets(k), continue; end
            h1       = lastHist;
            h2       = currentHists{k};
            histDist = sqrt(1 - sum(sqrt(h1 .* h2)));

            cost = histDist;
            
            distance = norm(currentCentroids(k,:) - tracks(t).centroid) ;
            if(distance > maxDist)
                cost = cost + 0.1*distance/maxDist; % Penalize if spatial distance is large
            end

            if cost < bestCost
                bestCost = cost;
                bestIdx   = k;
            end
        end

        if bestIdx > 0

            newCentroid = currentCentroids(bestIdx,:);
            tracks(t).centroid   = newCentroid;
            tracks(t).trajectory = [tracks(t).trajectory; newCentroid];
            tracks(t).hist       = 0.95 * tracks(t).hist + 0.05 * currentHists{bestIdx};
            tracks(t).bbox       = currentBBoxes{bestIdx};
            tracks(t).missedFrames = 0;
            tracks(t).confidence = 1.0 - bestCost; % Confidence based on histogram similarity
            tracks(t).isActive   = true;
            tracks(t).activeFrames = tracks(t).activeFrames + 1; % Increment active frame counter

            assignedDets(bestIdx)  = 1;
            assignedTracks(t)      = 1;

        end
    end
    % --- Update missed frames and deactivate tracks that have been lost for too long ---
    for t = 1:length(tracks)
        if ~assignedTracks(t)
            tracks(t).missedFrames = tracks(t).missedFrames + 1;
            if tracks(t).missedFrames > maxMissed
                tracks(t).trajectory = [];
                if tracks(t).missedFrames > 2 * tracks(t).activeFrames
                    tracks(t).isActive = false; % Deactivate track if it has been missed for too long compared to its active duration
                end
            end
        end
    end
    % --- Spawn new tracks for unmatched detections ---
    for k = 1:size(currentCentroids,1)
        if ~assignedDets(k)
            trackCount = trackCount + 1;

            tracks(end+1).id           = trackCount;
            tracks(end).centroid       = currentCentroids(k,:);
            tracks(end).trajectory     = currentCentroids(k,:);
            tracks(end).hist           = currentHists{k};
            tracks(end).bbox           = currentBBoxes{k};
            tracks(end).missedFrames   = 0;
            tracks(end).velocity       = [0 0];
            tracks(end).isActive       = true;
            tracks(end).confidence     = 1.0;
        end
    end

    %draw base image
    imshow(imgfr);

    hold on

    %draw ground truth boxes
    gt_frame = gt(gt(:,1) == frameNumber, :);

    for k = 1:size(gt_frame,1)
        x    = gt_frame(k,3);
        y    = gt_frame(k,4);
        w    = gt_frame(k,5);
        h    = gt_frame(k,6);
        conf = gt_frame(k,7);

        if conf > 0
            rectangle('Position', [x y w h], ...
                      'EdgeColor', 'g', ...
                      'LineWidth', 2);
        end
    end

    % draw tracked boxes and trajectories

    for t = 1:length(tracks)

        % Only draws active tracks
        if ~tracks(t).isActive || tracks(t).missedFrames > 0
            continue;
        end

        %draw blob bounding box
        bbox = tracks(t).bbox;
        rectangle('Position', bbox, ...
            'EdgeColor', 'y', ...
            'LineWidth', 2);


        % Write each tracked object to results file:
        % frame id left top width height confidence x y z
        
        if ~isempty(bbox)
            left  = bbox(1);
            top   = bbox(2);
            width = bbox(3);
            height= bbox(4);
            confidence = tracks(t).confidence;
            fprintf(outFile, '%d %d %.2f %.2f %.2f %.2f %.2f -1 -1 -1\n', ...
                    frameNumber, tracks(t).id, left, top, width, height, confidence);
        end

        traj = tracks(t).trajectory;

        % Draws trajectory
        if size(traj,1) > 1
            startIdx = max(1, size(traj,1) - maxTrail);
            plot(traj(startIdx:end,1), traj(startIdx:end,2), ...
                 'r-', 'LineWidth', 2);
        end

        % Draw ID at current position
        cx = tracks(t).centroid(1);
        cy = tracks(t).centroid(2);

        text(cx, cy, sprintf('ID %d', tracks(t).id), ...
            'Color', 'y', ...
            'FontSize', 10, ...
            'FontWeight', 'bold');


    end


    hold off;
    drawnow;
end

fclose(outFile);
disp('Completed algorithm.')
hold on;

function h = computeHistogram(img, bbox, blobMask)

    x   = round(bbox(1));
    y   = round(bbox(2));
    w   = round(bbox(3));
    hgt = round(bbox(4));

    % Clamp boundaries safely
    x   = max(1, x);
    y   = max(1, y);
    w   = min(size(img,2) - x + 1, w);
    hgt = min(size(img,1) - y + 1, hgt);

    if w < 1 || hgt < 1
        h = zeros(32,1);
        return;
    end

    % Extract image patch
    patch = img(y:y+hgt-1, x:x+w-1, :);

    % Resize mask if regionprops/Image and bbox dimensions differ slightly
    if size(blobMask,1) ~= size(patch,1) || size(blobMask,2) ~= size(patch,2)
        blobMask = imresize(blobMask, [size(patch,1), size(patch,2)], 'nearest');
    end

    hsvImg = rgb2hsv(patch);

    % Extract only foreground pixels
    hueVals = hsvImg(:,:,1);
    %satVals = hsvImg(:,:,2);

    hueVals = hueVals(blobMask);
    %satVals = satVals(blobMask);

    % If mask is empty, return zero histogram
    if isempty(hueVals)
        h = zeros(32,1);
        return;
    end

    % Compute histograms manually using selected pixels
    h1 = histcounts(hueVals, 32, 'BinLimits', [0 1], 'Normalization', 'count');
    %h2 = histcounts(satVals, 32, 'BinLimits', [0 1], 'Normalization', 'count');

    %h = [h1(:); h2(:)];
    h = h1;
    h = h / (sum(h) + eps);
end