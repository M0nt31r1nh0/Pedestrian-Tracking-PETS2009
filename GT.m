clear all; close all; 

gt = readmatrix('PETS-S2L1\gt\gt.txt');
numFramesBG = 794;

for i = 1:numFramesBG
    img = imread(sprintf('Crowd_PETS\\S2\\L1\\Time_12-34\\View_001\\frame_%.4d.jpg', i));

    if i == 1
        [h, w, c] = size(img);
        stack = zeros(h, w, c, numFramesBG, 'double');
    end

    stack(:,:,:,i) = double(img);
end

imgbk = uint8(median(stack, 4));

thr      = 40;
minArea  = 100;
baseValue = 793;

se = strel('disk', 2);

maxDist    = 50;
trackCount = 0;
maxTrail   = 20;
maxMissed  = 5;   

tracks = struct( ...
    'id',           {}, ...
    'centroid',     {}, ...
    'trajectory',   {}, ...
    'hist',         {}, ...
    'missedFrames', {} ...   
);

figure;
for i = 0:baseValue
    frameNumber = i + 1;
    imgfr = imread(sprintf('Crowd_PETS\\S2\\L1\\Time_12-34\\View_001\\frame_%.4d.jpg', frameNumber));
    imgdif = (abs(double(imgbk(:,:,1)) - double(imgfr(:,:,1))) > thr) | ...
             (abs(double(imgbk(:,:,2)) - double(imgfr(:,:,2))) > thr) | ...
             (abs(double(imgbk(:,:,3)) - double(imgfr(:,:,3))) > thr);

    bw1 = imgdif;
    bw2 = imclose(bw1, se);
    bw3 = imerode(bw2, se);
    bwR = bw3;

    [lb, num] = bwlabel(bwR);
    stats = regionprops(lb, 'Area', 'BoundingBox');
    inds = find([stats.Area] > minArea);

    imshow(imgfr);
    hold on

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

    currentCentroids = [];
    currentHists     = {};
    currentBBoxes    = {};

    for idx = 1:length(inds)
        j    = inds(idx);
        bbox = stats(j).BoundingBox;

        rectangle('Position', bbox, ...
                  'EdgeColor', 'y', ...
                  'LineWidth', 2);

        cx = bbox(1) + bbox(3)/2;
        cy = bbox(2) + bbox(4)/2;

        hist_val = computeHistogram(imgfr, bbox);

        currentCentroids          = [currentCentroids; cx cy];
        currentHists{end+1}       = hist_val;
        currentBBoxes{end+1}      = bbox;
    end

    assignedDets   = zeros(size(currentCentroids,1), 1);
    assignedTracks = zeros(length(tracks), 1);

    for t = 1:length(tracks)

        lastPos  = tracks(t).centroid;
        lastHist = tracks(t).hist;

        bestScore = inf;
        bestIdx   = -1;

        for k = 1:size(currentCentroids,1)

            if assignedDets(k), continue; end

            dist = norm(currentCentroids(k,:) - lastPos);
            if dist > maxDist, continue; end

            h1       = lastHist;
            h2       = currentHists{k};
            histDist = sqrt(1 - sum(sqrt(h1 .* h2)));

            cost = dist + 50 * histDist;

            if cost < bestScore
                bestScore = cost;
                bestIdx   = k;
            end
        end

        if bestIdx > 0
            tracks(t).centroid     = currentCentroids(bestIdx,:);
            tracks(t).trajectory   = [tracks(t).trajectory; currentCentroids(bestIdx,:)];
            tracks(t).hist         = 0.7 * tracks(t).hist + 0.3 * currentHists{bestIdx};
            tracks(t).missedFrames = 0;        
            assignedDets(bestIdx)  = 1;
            assignedTracks(t)      = 1;
        end
    end

    % --- Increment miss counter for unmatched tracks ---
    for t = 1:length(tracks)
        if ~assignedTracks(t)
            tracks(t).missedFrames = tracks(t).missedFrames + 1;
        end
    end

    % --- Prune dead tracks 
    if ~isempty(tracks)
        tracks = tracks([tracks.missedFrames] <= maxMissed);
    end

    % --- Spawn new tracks for unmatched detections ---
    for k = 1:size(currentCentroids,1)
        if ~assignedDets(k)
            trackCount = trackCount + 1;

            tracks(end+1).id           = trackCount;
            tracks(end).centroid       = currentCentroids(k,:);
            tracks(end).trajectory     = currentCentroids(k,:);
            tracks(end).hist           = currentHists{k};
            tracks(end).missedFrames   = 0;
        end
    end

    for t = 1:length(tracks)

        % Only draws active tracks
        if tracks(t).missedFrames > 0
            continue;   
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

function h = computeHistogram(img, bbox)

    x   = round(bbox(1));
    y   = round(bbox(2));
    w   = round(bbox(3));
    hgt = round(bbox(4));

    % Clamp boundaries
    x   = max(1, x);
    y   = max(1, y);
    w   = min(size(img,2) - x, w);
    hgt = min(size(img,1) - y, hgt);

    patch = img(y:y+hgt-1, x:x+w-1, :);

    hsvImg = rgb2hsv(patch);

    % Use Hue channel (32 bins)
    h = imhist(hsvImg(:,:,1), 32);

    % Normalize
    h = h / (sum(h) + eps);
end