clc; clear; close all;

%% =========================
% 1) LOAD DATA
% ==========================
filename = 'results.txt';

% Columns:
% 1 frame, 2 id, 3 left, 4 top, 5 width, 6 height, 7 conf, 8 x3D, 9 y3D, 10 z3D
data = readmatrix(filename);

% Keep only valid detections
data = data(data(:,7) > 0, :);

% Sort by pedestrian ID and frame
data = sortrows(data, [2 1]);

ids = unique(data(:,2));

%% =========================
% 2) PARAMETERS
% ==========================
M = 11;          % number of resampled points per trajectory
minLen = 11;      % minimum number of detections required
Kmax = 20;        % maximum number of clusters to test

% Choose trajectory representation:
% mode = 1 -> keep original image coordinates (scene-flow clustering)
% mode = 2 -> normalize to start point (shape-only clustering)
mode = 2;

%% =========================
% 3) BUILD TRAJECTORY-SHAPE FEATURES
% ==========================
X = [];              % feature matrix
validIDs = [];       % kept pedestrian IDs
rawTraj = {};        % original trajectories for plotting
resampledTraj = {};  % resampled trajectories

for i = 1:length(ids)
    id = ids(i);
    traj = data(data(:,2) == id, :);

    if size(traj,1) < minLen
        continue;
    end

    left = traj(:,3);
    top  = traj(:,4);
    w    = traj(:,5);
    h    = traj(:,6);

    % Bounding box center
    cx = left + w/2;
    cy = top  + h/2;

    % Optional smoothing to reduce jitter
    cx = smoothdata(cx, 'movmean', 3);
    cy = smoothdata(cy, 'movmean', 3);

    % Original sample positions
    tOriginal = linspace(0,1,length(cx));

    % Resample to fixed number of points
    tNew = linspace(0,1,M);
    cx_res = interp1(tOriginal, cx, tNew, 'linear');
    cy_res = interp1(tOriginal, cy, tNew, 'linear');

    % Representation choice
    switch mode
        case 1
            % Keep original image coordinates
            featX = cx_res;
            featY = cy_res;

        case 2
            % Normalize by starting point (shape-only)
            featX = cx_res - cx_res(1);
            featY = cy_res - cy_res(1);

            % Optional scale normalization
            pathLen = sum(sqrt(diff(featX).^2 + diff(featY).^2));
            if pathLen > 0
                featX = featX / pathLen;
                featY = featY / pathLen;
            end
    end

    % Flatten into one vector: [x1 y1 x2 y2 ... xM yM]
    feat = reshape([featX; featY], 1, []);

    X = [X; feat];
    validIDs = [validIDs; id];
    rawTraj{end+1} = [cx cy];
    resampledTraj{end+1} = [cx_res(:) cy_res(:)];
end

fprintf('Valid trajectories used: %d\n', size(X,1));

%% =========================
% 4) STANDARDIZE FEATURES
% ==========================
Xz = zscore(X);

%% =========================
% 5) SELECT NUMBER OF CLUSTERS WITH BIC
% ==========================
GMModels = cell(Kmax,1);
BIC = inf(Kmax,1);

options = statset('MaxIter',10000, 'Display','final');

for k = 1:Kmax
    try
        GMModels{k} = fitgmdist(Xz, k, ...
            'CovarianceType', 'full', ...
            'RegularizationValue', 1e-4, ...
            'Replicates', 100, ...
            'Options', options);

        BIC(k) = GMModels{k}.BIC;
    catch
        warning('GMM failed for K = %d', k);
    end
end

[~, bestK] = min(BIC);
gmm = GMModels{bestK};

fprintf('Best number of clusters according to BIC: %d\n', bestK);

%% =========================
% 6) EM CLUSTERING
% ==========================
[idx, nlogL, P] = cluster(gmm, Xz);

fprintf('Final negative log-likelihood: %.4f\n', nlogL);

%% =========================
% 7) BIC CURVE
% ==========================
figure;
plot(1:Kmax, BIC, '-o', 'LineWidth', 2);
xlabel('Number of Clusters K');
ylabel('BIC');
title('Model Selection using BIC');
grid on;
savefig( "BIC_CURVE.fig" )
%% =========================
% 8) PLOT RESAMPLED TRAJECTORIES BY CLUSTER
% ==========================
figure; hold on;
title(sprintf('Trajectory-Shape EM Clustering (K = %d)', bestK));
xlabel('Image X');
ylabel('Image Y');
set(gca, 'YDir', 'reverse');
grid on;

colors = lines(bestK);

for i = 1:length(resampledTraj)
    tr = resampledTraj{i};
    plot(tr(:,1), tr(:,2), '-', 'Color', colors(idx(i),:), 'LineWidth', 1.2);
    plot(tr(1,1), tr(1,2), 'o', 'Color', colors(idx(i),:), 'MarkerFaceColor', colors(idx(i),:));
end
savefig( "CLUSTER_SHAPES.fig" )
%% =========================
% 9) COMPUTE AND PLOT MEAN TRAJECTORY OF EACH CLUSTER
% ==========================
figure; hold on;
title('Mean Trajectory Shape per Cluster');
xlabel('Image X');
ylabel('Image Y');
set(gca, 'YDir', 'reverse');
grid on;

for k = 1:bestK
    clusterX = X(idx == k, :);

    if isempty(clusterX)
        continue;
    end

    meanVec = mean(clusterX, 1);

    % Recover trajectory points
    meanX = meanVec(1:2:end);
    meanY = meanVec(2:2:end);

    plot(meanX, meanY, '-', 'Color', colors(k,:), 'LineWidth', 3);
    plot(meanX(1), meanY(1), 's', 'Color', colors(k,:), 'MarkerFaceColor', colors(k,:), 'MarkerSize', 8);
    text(meanX(round(end/2)), meanY(round(end/2)), sprintf('Cluster %d', k), ...
        'FontWeight', 'bold');
end

savefig( "CLUSTER_MEANS.fig" )
%% =========================
% 10) CLUSTER STATISTICS
% ==========================
fid = fopen('cluster_statistics.txt', 'w');
fprintf(fid, '\n===== TRAJECTORY-SHAPE CLUSTER STATISTICS =====\n');

for k = 1:bestK
    members = find(idx == k);
    Nk = numel(members);

    if Nk == 0
        continue;
    end

    durations = zeros(Nk,1);
    pathLengths = zeros(Nk,1);
    netDisplacements = zeros(Nk,1);

    for j = 1:Nk
        tr = rawTraj{members(j)};
        dx = diff(tr(:,1));
        dy = diff(tr(:,2));

        pathLengths(j) = sum(sqrt(dx.^2 + dy.^2));
        netDisplacements(j) = sqrt((tr(end,1)-tr(1,1))^2 + (tr(end,2)-tr(1,2))^2);
        durations(j) = size(tr,1);
    end

    fprintf(fid, '\nCluster %d\n', k);
    fprintf(fid, '  Number of trajectories: %d\n', Nk);
    fprintf(fid, '  Mean duration: %.2f frames\n', mean(durations));
    fprintf(fid, '  Mean path length: %.2f px\n', mean(pathLengths));
    fprintf(fid, '  Mean net displacement: %.2f px\n', mean(netDisplacements));
end

fclose(fid);
fprintf('Cluster statistics saved to cluster_statistics.txt\n');

%% =========================
% 11) POSTERIOR CONFIDENCE ANALYSIS
% ==========================
maxProb = max(P, [], 2);

figure;
histogram(maxProb, 10);
xlabel('Maximum Posterior Probability');
ylabel('Number of Trajectories');
title('EM Confidence of Trajectory Assignments');
grid on;

fprintf('\n===== UNCERTAIN TRAJECTORIES =====\n');
ambiguous = find(maxProb < 0.7);

for i = 1:length(ambiguous)
    fprintf('Trajectory ID %d is ambiguous (max posterior = %.3f)\n', ...
        validIDs(ambiguous(i)), maxProb(ambiguous(i)));
end
savefig( "CONFIDENCE.fig" )
%% =========================
% 12) SAVE RESULTS
% ==========================
resultsTable = table(validIDs, idx, maxProb, ...
    'VariableNames', {'ID','Cluster','Confidence'});

writetable(resultsTable, 'EM_results.csv');
fprintf('\nSaved results to EM_results.csv\n');