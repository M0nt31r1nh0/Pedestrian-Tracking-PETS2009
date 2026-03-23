clear all; close all;

gt = readmatrix('PETS-S2L1\\gt\\gt.txt');  %read the ground truth

numFramesBG = 794; %better results with 794, but faster with lower number of frames to do the median

for i = 1:numFramesBG
    img = imread(sprintf('Crowd_PETS\\S2\\L1\\Time_12-34\\View_001\\frame_%.4d.jpg', i));

    if i == 1
        [h, w, c] = size(img);
        stack = zeros(h, w, c, numFramesBG, 'double');
    end

    stack(:, :, :, i) = double(img);
end

imgbk = uint8(median(stack, 4));

thr = 40;
minArea = 100;
baseValue = 794;

se = strel('disk', 2);

maxDist = 50;        
maxMissed = 10;      
maxTrail = 20;       

tracks = struct('id', {}, ...
                'bbox', {}, ...
                'centroid', {}, ...
                'trajectory', {}, ...
                'missed', {});

nextID = 1;

% Clear old results file before writing new results
if exist('results.txt', 'file')
    delete('results.txt');
end

% Open a file to save detection/tracking results
outFile = fopen('results.txt', 'w');
if outFile == -1
    error('Cannot open output file results.txt for writing.');
end

figure; hold on

for i = 1: baseValue
    imgfr = imread(sprintf('Crowd_PETS\\S2\\L1\\Time_12-34\\View_001\\frame_%.4d.jpg', i));
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

    [H, W, ~] = size(imgfr);
    borderMargin = 5;
    
    validInds = [];
    
    for k = inds
        bbox = regionProps(k).BoundingBox;
    
        x = bbox(1);
        y = bbox(2);
        w = bbox(3);
        h = bbox(4);
    
        if x > borderMargin && ...
           y > borderMargin && ...
           (x + w) < (W - borderMargin) && ...
           (y + h) < (H - borderMargin)
    
            validInds = [validInds k];
        end
    end
    
    inds = validInds;

    detections = struct('bbox', {}, 'centroid', {});
    for k = 1: length(inds)
        bbox = regionProps(inds(k)).BoundingBox;
        cx = bbox(1) + bbox(3)/2;
        cy = bbox(2) + bbox(4)/2;

        detections(k).bbox = bbox;
        detections(k).centroid = [cx cy];
    end

    nTracks = length(tracks);
    nDetections = length(detections);

    assignedTracks = zeros(1, nTracks);
    assignedDetections = zeros(1, nDetections);

    if nTracks > 0 && nDetections > 0
        D = inf(nTracks, nDetections);

        for t = 1:nTracks
            for d = 1:nDetections
                D(t,d) = norm(tracks(t).centroid - detections(d).centroid);
            end
        end

        while true
            [minVal, linearIdx] = min(D(:));

            if isempty(minVal) || isinf(minVal) || minVal > maxDist
                break;
            end

            [t, d] = ind2sub(size(D), linearIdx);

            if ~assignedTracks(t) && ~assignedDetections(d)
               
                tracks(t).bbox = detections(d).bbox;
                tracks(t).centroid = detections(d).centroid;
                tracks(t).trajectory = [tracks(t).trajectory; detections(d).centroid];
                tracks(t).missed = 0;

                assignedTracks(t) = 1;
                assignedDetections(d) = 1;
            end

            D(t,:) = inf;
            D(:,d) = inf;
        end
    end

    for t = 1:length(tracks)
        if ~assignedTracks(t)
            tracks(t).missed = tracks(t).missed + 1;
        end
    end

    keepIdx = [tracks.missed] <= maxMissed;
    tracks = tracks(keepIdx);

    for d = 1: nDetections
        if ~assignedDetections(d)
            tracks(end+1).id = nextID;
            tracks(end).bbox = detections(d).bbox;
            tracks(end).centroid = detections(d).centroid;
            tracks(end).trajectory = detections(d).centroid;
            tracks(end).missed = 0;
            nextID = nextID + 1;
        end
    end

    imshow(imgfr);
    hold on;

    gt_frame = gt(gt(:,1) == i, :);

    for k = 1: size(gt_frame,1)
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

    for t = 1: length(tracks)
        if tracks(t).missed == 0
            bbox = tracks(t).bbox;
            cx = tracks(t).centroid(1);
            cy = tracks(t).centroid(2);

            rectangle('Position', bbox, ...
                      'EdgeColor', 'y', ...
                      'LineWidth', 2);

            text(cx, cy, sprintf('%d', tracks(t).id), ...
                'Color', 'y', ...
                'FontSize', 10, ...
                'FontWeight', 'bold');

            traj = tracks(t).trajectory;
            if size(traj,1) > 1
                startIdx = max(1, size(traj,1) - maxTrail);
                plot(traj(startIdx:end,1), traj(startIdx:end,2), ...
                     'r-', 'LineWidth', 2);
            end

            % Write each tracked object to results file:
            % frame id left top width height confidence x y z
            left = bbox(1);
            top  = bbox(2);
            width = bbox(3);
            height = bbox(4);
            confidence = 1.0;
            fprintf(outFile, '%d %d %.2f %.2f %.2f %.2f %.2f -1 -1 -1\n', ...
                    i, tracks(t).id, left, top, width, height, confidence);
        end
    end

    drawnow;
end

fclose(outFile);


