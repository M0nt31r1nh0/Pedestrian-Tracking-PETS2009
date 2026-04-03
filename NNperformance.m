clear; close all; clc;

%% =========================
%  SETTINGS
% ==========================
gtFile      = 'PETS-S2L1\gt\gt.txt';
resultsFile = 'NNresults.txt';
imgFolder   = 'Crowd_PETS\S2\L1\Time_12-34\View_001';

iouMatchThreshold = 0.5;       % threshold for FP/FN stats
successThresholds = 0:0.1:1;  % for success plot

saveErrorFrames = true;
maxSavedExamples = 0;%only needed to save frames for the read.me
outputFolder = 'NNevaluation_output';

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

%% =========================
%  LOAD DATA
% ==========================
gt = readmatrix(gtFile);
res = readmatrix(resultsFile);

% GT format assumed:
% [frame id x y w h conf ...]
gt = gt(gt(:,7) > 0, :);   % only valid GT boxes

allFrames = unique([gt(:,1); res(:,1)]);
numFrames = max(allFrames);

%% =========================
%  GLOBAL COUNTERS
% ==========================
totalGT   = 0;
totalPred = 0;
totalTP   = 0;
totalFP   = 0;
totalFN   = 0;

% For success plot
successRates = zeros(length(successThresholds),1);

% Store frame-level error info
frameFP = zeros(numFrames,1);
frameFN = zeros(numFrames,1);

% Store example frames
errorFrames = [];

%% =========================
%  MAIN EVALUATION
% ==========================
for f = 1:numFrames

    % Ground truth for frame
    gt_frame = gt(gt(:,1) == f, :);
    gt_boxes = gt_frame(:,3:6);  % [x y w h]

    % Predictions for frame
    res_frame = res(res(:,1) == f, :);
    pred_boxes = res_frame(:,3:6);  % [x y w h]

    nGT   = size(gt_boxes,1);
    nPred = size(pred_boxes,1);

    totalGT   = totalGT + nGT;
    totalPred = totalPred + nPred;

    % Evaluate at main threshold (for FP/FN/TP stats)
    [matchedGT, matchedPred, iouMatrix] = matchBoxes(gt_boxes, pred_boxes, iouMatchThreshold);

    TP = length(matchedGT);
    FP = nPred - TP;
    FN = nGT - TP;

    totalTP = totalTP + TP;
    totalFP = totalFP + FP;
    totalFN = totalFN + FN;

    frameFP(f) = FP;
    frameFN(f) = FN;

    if FP > 0 || FN > 0
        errorFrames(end+1) = f; %#ok<SAGROW>
    end

    % Success plot accumulation
    for t = 1:length(successThresholds)
        thr = successThresholds(t);
        [matchedGT_thr, ~, ~] = matchBoxes(gt_boxes, pred_boxes, thr);

        % success = matched GT / total GT in frame
        if nGT > 0
            successRates(t) = successRates(t) + length(matchedGT_thr);
        end
    end
end

% Normalize success rates
if totalGT > 0
    successRates = successRates / totalGT;
end

%% =========================
%  FINAL METRICS
% ==========================
FN_percent_GT   = 100 * totalFN / max(totalGT,1);
FP_percent_pred = 100 * totalFP / max(totalPred,1);
FP_percent_GT   = 100 * totalFP / max(totalGT,1);

precision = totalTP / max(totalTP + totalFP,1);
recall    = totalTP / max(totalTP + totalFN,1);
f1score   = 2 * precision * recall / max(precision + recall, eps);

fprintf('\n========== EVALUATION ==========\n');
fprintf('Total GT objects:        %d\n', totalGT);
fprintf('Total predicted objects: %d\n', totalPred);
fprintf('True Positives (TP):     %d\n', totalTP);
fprintf('False Positives (FP):    %d\n', totalFP);
fprintf('False Negatives (FN):    %d\n', totalFN);
fprintf('\n');
fprintf('FN %% (over GT):          %.2f%%\n', FN_percent_GT);
fprintf('FP %% (over predictions): %.2f%%\n', FP_percent_pred);
fprintf('FP %% (over GT):          %.2f%%\n', FP_percent_GT);
fprintf('\n');
fprintf('Precision:               %.4f\n', precision);
fprintf('Recall:                  %.4f\n', recall);
fprintf('F1-score:                %.4f\n', f1score);

%% =========================
%  SAVE SUMMARY TEXT FILE
% ==========================
summaryFile = fullfile(outputFolder, 'evaluation_summary.txt');
fid = fopen(summaryFile, 'w');

fprintf(fid, '========== EVALUATION ==========\n');
fprintf(fid, 'Total GT objects:        %d\n', totalGT);
fprintf(fid, 'Total predicted objects: %d\n', totalPred);
fprintf(fid, 'True Positives (TP):     %d\n', totalTP);
fprintf(fid, 'False Positives (FP):    %d\n', totalFP);
fprintf(fid, 'False Negatives (FN):    %d\n', totalFN);
fprintf(fid, '\n');
fprintf(fid, 'FN %% (over GT):          %.2f%%\n', FN_percent_GT);
fprintf(fid, 'FP %% (over predictions): %.2f%%\n', FP_percent_pred);
fprintf(fid, 'FP %% (over GT):          %.2f%%\n', FP_percent_GT);
fprintf(fid, '\n');
fprintf(fid, 'Precision:               %.4f\n', precision);
fprintf(fid, 'Recall:                  %.4f\n', recall);
fprintf(fid, 'F1-score:                %.4f\n', f1score);

fclose(fid);

%% =========================
%  SUCCESS PLOT
% ==========================
figure;
plot(successThresholds, successRates, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('IoU Threshold');
ylabel('Success Rate');
title('Success Plot');
grid on;
ylim([0 1]);

saveas(gcf, fullfile(outputFolder, 'success_plot.png'));

%% =========================
%  FP / FN BAR CHART
% ==========================
figure;
bar([totalFP totalFN totalTP]);
set(gca, 'XTickLabel', {'False Positives', 'False Negatives', 'True Positives'});
ylabel('Count');
title('FP and FN Summary');
grid on;

saveas(gcf, fullfile(outputFolder, 'fp_fn_summary.png'));

%% =========================
%  FP/FN PER FRAME PLOT
% ==========================
figure;
plot(1:numFrames, frameFP, 'r', 'LineWidth', 1.5); hold on;
plot(1:numFrames, frameFN, 'm', 'LineWidth', 1.5);
xlabel('Frame');
ylabel('Count');
legend('FP per frame', 'FN per frame');
title('False Positives and False Negatives per Frame');
grid on;

saveas(gcf, fullfile(outputFolder, 'fp_fn_per_frame.png'));

%% =========================
%  SAVE EXAMPLE ERROR FRAMES
% ==========================
if saveErrorFrames
    numToSave = min(maxSavedExamples, length(errorFrames));

    for idx = 1:numToSave
        f = errorFrames(idx);

        imgPath = fullfile(imgFolder, sprintf('frame_%.4d.jpg', f-1)); % adjust if needed
        if ~exist(imgPath, 'file')
            imgPath = fullfile(imgFolder, sprintf('frame_%.4d.jpg', f)); % fallback
        end

        img = imread(imgPath);

        gt_frame = gt(gt(:,1) == f, :);
        gt_boxes = gt_frame(:,3:6);

        res_frame = res(res(:,1) == f, :);
        pred_boxes = res_frame(:,3:6);

        [matchedGT, matchedPred, ~] = matchBoxes(gt_boxes, pred_boxes, iouMatchThreshold);

        unmatchedGT   = setdiff(1:size(gt_boxes,1), matchedGT);
        unmatchedPred = setdiff(1:size(pred_boxes,1), matchedPred);

        figure('Visible', 'off');
        imshow(img); hold on;
        title(sprintf('Frame %d - Error Visualization', f));

        % Draw GT matched = green
        for g = matchedGT
            rectangle('Position', gt_boxes(g,:), 'EdgeColor', 'g', 'LineWidth', 2);
        end

        % Draw FN GT = magenta
        for g = unmatchedGT
            rectangle('Position', gt_boxes(g,:), 'EdgeColor', 'm', 'LineWidth', 2, 'LineStyle', '--');
            text(gt_boxes(g,1), gt_boxes(g,2)-5, 'FN', 'Color', 'm', 'FontWeight', 'bold');
        end

        % Draw FP predictions = red
        for p = unmatchedPred
            rectangle('Position', pred_boxes(p,:), 'EdgeColor', 'r', 'LineWidth', 2, 'LineStyle', '--');
            text(pred_boxes(p,1), pred_boxes(p,2)-5, 'FP', 'Color', 'r', 'FontWeight', 'bold');
        end

        % Draw matched predictions = cyan
        for p = matchedPred
            rectangle('Position', pred_boxes(p,:), 'EdgeColor', 'c', 'LineWidth', 1.5);
        end

        saveas(gcf, fullfile(outputFolder, sprintf('error_frame_%04d.png', f)));
        close;
    end
end

disp('Evaluation complete. Results saved to evaluation_output/');

%helper functions
function [matchedGT, matchedPred, iouMatrix] = matchBoxes(gt_boxes, pred_boxes, iouThreshold)

    nGT = size(gt_boxes,1);
    nPred = size(pred_boxes,1);

    matchedGT = [];
    matchedPred = [];

    if nGT == 0 || nPred == 0
        iouMatrix = zeros(nGT, nPred);
        return;
    end

    iouMatrix = zeros(nGT, nPred);

    for i = 1:nGT
        for j = 1:nPred
            iouMatrix(i,j) = computeIoU(gt_boxes(i,:), pred_boxes(j,:));
        end
    end

    while true
        [maxVal, linearIdx] = max(iouMatrix(:));

        if isempty(maxVal) || maxVal < iouThreshold
            break;
        end

        [g, p] = ind2sub(size(iouMatrix), linearIdx);

        matchedGT(end+1) = g; %#ok<AGROW>
        matchedPred(end+1) = p; %#ok<AGROW>

        iouMatrix(g,:) = -1;
        iouMatrix(:,p) = -1;
    end
end

function iou = computeIoU(areaA, areaB)
    % box = [x y w h]

    xA1 = areaA(1);
    yA1 = areaA(2);
    xA2 = areaA(1) + areaA(3);
    yA2 = areaA(2) + areaA(4);

    xB1 = areaB(1);
    yB1 = areaB(2);
    xB2 = areaB(1) + areaB(3);
    yB2 = areaB(2) + areaB(4);

    interX1 = max(xA1, xB1);
    interY1 = max(yA1, yB1);
    interX2 = min(xA2, xB2);
    interY2 = min(yA2, yB2);

    interW = max(0, interX2 - interX1);
    interH = max(0, interY2 - interY1);
    interArea = interW * interH;

    areaA = areaA(3) * areaA(4);
    areaB = areaB(3) * areaB(4);

    iou = interArea / max(areaA + areaB - interArea, eps);
end