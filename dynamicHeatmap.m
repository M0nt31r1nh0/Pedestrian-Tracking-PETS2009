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

unique_frames = unique(frame);

temp_map = zeros(img_height, img_width);
temp_map_smooth = zeros(img_height, img_width);

for f = 1:length(unique_frames)
    current_frame = unique_frames(f);
    
    idx = frame == current_frame;
    
    for i = find(idx)'
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
            
            temp_map(y_start:y_end, x_start:x_end) = temp_map(y_start:y_end, x_start:x_end) + h(ky_start:ky_end, kx_start:kx_end);
        end
    end
    
    temp_map_smooth = temp_map;
    if max(temp_map(:)) > 0
        temp_map_smooth = temp_map_smooth / max(temp_map(:));
    end
    
    imagesc(temp_map_smooth);
    axis image;
    colormap('hot');
    colorbar;
    title(['Frame: ', num2str(current_frame)]);
    
    drawnow;
end

figure;
imagesc(temp_map_smooth);
axis image;
colormap('hot');
colorbar;
title('Static Occupancy Heatmap');