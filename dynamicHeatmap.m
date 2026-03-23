data = readmatrix('PETS-S2L1\\det\\det.txt');

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


unique_frames = unique(frame);

temp_map = zeros(img_height, img_width);
temp_map_smooth = zeros(img_height, img_width);

for f = 1:length(unique_frames)
    current_frame = unique_frames(f);
    
    idx = frame == current_frame;
    
    for i = find(idx)'
        x = round(cx(i));
        y = round(cy(i));
        
        if x > 0 && x <= img_width && y > 0 && y <= img_height
            temp_map(y, x) = temp_map(y, x) + 1;
        end
    end
    
    temp_map_smooth = imgaussfilt(temp_map, 10);
    temp_map_smooth = temp_map_smooth / max(temp_map_smooth(:));
    
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