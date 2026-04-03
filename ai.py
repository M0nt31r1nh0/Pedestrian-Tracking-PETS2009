import os
import re
from glob import glob
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Input folder containing frames
input_folder = 'Crowd_PETS/S2/L1/Time_12-34/View_001'

# Output file
output_file = 'NNresults.txt'

# Initialize output file with optional header comment
with open(output_file, 'w') as f:
    f.write('# frame id left top width height conf x y z\n')

# Find image files (jpg, png, etc.) and sort by filename
img_files = sorted(glob(os.path.join(input_folder, '*.*')))
img_files = [p for p in img_files if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]

if not img_files:
    raise RuntimeError(f'No image files found in {input_folder}')

with open(output_file, 'a') as f:
    for img_path in img_files:
        # Derive frame number from filename 'frame_0000.jpg' -> 0
        basename = os.path.basename(img_path)
        frame_match = re.search(r'frame_(\d{4})', basename)
        if frame_match:
            frame_no = int(frame_match.group(1))
        else:
            # fallback to all digits
            frame_match = re.search(r'([0-9]+)', basename)
            frame_no = int(frame_match.group(1)) if frame_match else (img_files.index(img_path) + 1)

        results = model(img_path)

        for result in results:
            names = [result.names[int(cls)] for cls in result.boxes.cls]
            boxes = result.boxes

            # id: detector outputs set to -1
            pid = -1

            for i in range(boxes.shape[0]):
                if names[i] != 'person':
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                width = x2 - x1
                height = y2 - y1
                conf = float(boxes.conf[i])

                # 3D coordinates not available
                x3 = -1
                y3 = -1
                z3 = -1

                f.write(f"{frame_no+1} {pid} {x1:.2f} {y1:.2f} {width:.2f} {height:.2f} {conf:.4f} {x3} {y3} {z3}\n")

print(f'Done: results written to {output_file} ({len(img_files)} images)')