data = readmatrix('results.txt');

frame = data(:,1);
id    = data(:,2);
left  = data(:,3);
top   = data(:,4);
width = data(:,5);
height= data(:,6);
conf  = data(:,7);

% world coordinates, not implemented
x3d = data(:,8);
y3d = data(:,9);
z3d = data(:,10);


cx = left + width / 2;
cy = top  + height / 2;


img_width = 768;
img_height = 576;

hm = zeros(img_height, img_width);

for i = 1:length(cx)
    x = round(cx(i));
    y = round(cy(i));
    
    if x > 0 && x <= img_width && y > 0 && y <= img_height
        hm(y, x) = hm(y, x) + 1;
    end
end

sigma = 15; % controls spread
heatmap_smooth = imgaussfilt(hm, sigma);
heatmap_smooth = heatmap_smooth / max(heatmap_smooth(:));

figure;
imagesc(heatmap_smooth);
axis image;
colormap('hot');
colorbar;
title('Static Occupancy Heatmap');