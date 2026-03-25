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

sigma = 10;
kernel_size = 61;
h = fspecial('gaussian', kernel_size, sigma);
half_k = floor(kernel_size / 2);

hm = zeros(img_height, img_width);

for i = 1:length(cx)
    x_int = round(cx(i));
    y_int = round(cy(i));
    
    if x_int > 0 && x_int <= img_width && y_int > 0 && y_int <= img_height
        y_start = max(1, y_int - half_k);
        y_end = min(img_height, y_int + half_k);
        x_start = max(1, x_int - half_k);
        x_end = min(img_width, x_int + half_k);
        
        ky_start = 1 + (y_start - (y_int - half_k));
        ky_end = ky_start + (y_end - y_start);
        kx_start = 1 + (x_start - (x_int - half_k));
        kx_end = kx_start + (x_end - x_start);
        
        hm(y_start:y_end, x_start:x_end) = hm(y_start:y_end, x_start:x_end) + h(ky_start:ky_end, kx_start:kx_end);
    end
end



if max(hm(:)) > 0
    hm = hm / max(hm(:));
end

figure;
imagesc(hm);
axis image;
colormap('hot');
colorbar;
title('Static Occupancy Heatmap');