# generate_edges.py

import cv2
import glob
import os

# Define input and output folders
target_folder = "data/Aerovision_data/train_B"
edge_map_folder = "data/Aerovision_data/train_edges"
os.makedirs(edge_map_folder, exist_ok=True)

# Process every image in the target folder
for img_path in glob.glob(os.path.join(target_folder, "*.png")):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Use the Canny edge detector to find the edges
    # You can experiment with the threshold values (100, 200) for best results
    edges = cv2.Canny(img, threshold1=100, threshold2=200)

    filename = os.path.basename(img_path)
    cv2.imwrite(os.path.join(edge_map_folder, filename), edges)

print("✅ Edge maps generated successfully!")